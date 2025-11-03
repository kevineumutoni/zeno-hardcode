import os
from rest_framework import viewsets, permissions, status, generics
from rest_framework.response import Response
from rest_framework.decorators import action
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.contrib.auth.tokens import default_token_generator
from django.conf import settings
from django.core.mail import send_mail
from rest_framework.authtoken.models import Token
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.exceptions import PermissionDenied
from django.contrib.auth import authenticate
from users.models import User, Review
from agents.models import Agent, Tool
from runs.models import Run, RunInputFile, RunOutputArtifact
from conversations.models import Conversation, Step
from .serializers import PasswordResetRequestSerializer, PasswordResetConfirmSerializer
from .serializers import (
    UserSerializer, ReviewSerializer, AgentSerializer, ConversationSerializer,
    ToolSerializer, StepSerializer, RunInputFileSerializer, RunOutputArtifactSerializer,
    RunSerializer, ConversationWithRunsSerializer
)
from .permissions import IsAdmin
import threading
import requests
from django.utils import timezone
from datetime import timedelta
from rest_framework.views import APIView
import tempfile
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import csv
import pandas as pd
from docx import Document

ZEN_AGENT_API_URL = os.environ.get("ZEN_AGENT_API_URL")
MAX_CONVERSATIONS_PER_DAY = 5
MAX_RUNS_PER_CONVERSATION_PER_DAY = 20


def determine_file_type(file_obj):
    """Determines the simplified file type for the database."""
    file_name = file_obj.name.lower()
    content_type = file_obj.content_type

    if content_type == "application/pdf" or file_name.endswith(".pdf"):
        return 'pdf'
    elif content_type.startswith("image/") or file_name.endswith((".png", ".jpg", ".jpeg")):
        return 'image'
    elif file_name.endswith(".csv"):
        return 'csv'
    elif file_name.endswith((".xlsx", ".xls", ".xlsm")):
        return 'excel'
    elif file_name.endswith((".docx", ".doc")):
        return 'word'
    elif file_name.endswith(".txt") or content_type.startswith("text/"):
        return 'text'
    return 'text' 

def extract_text_from_file(file_obj):
    """Extract text from PDF, image, CSV, Excel, Word, or plain text."""
    try:
        if file_obj.size > 50 * 2024 * 2024:  
            return "[File too large for processing]"

        file_name = file_obj.name.lower()
        content_type = file_obj.content_type

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            for chunk in file_obj.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        text = ""

        
        if content_type == "application/pdf" or file_name.endswith(".pdf"):
            reader = PdfReader(tmp_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)

       
        elif content_type.startswith("image/") or file_name.endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(tmp_path)
            text = pytesseract.image_to_string(image)

        
        elif file_name.endswith(".csv"):
            df = pd.read_csv(tmp_path)
            text = df.to_string(index=False)

        
        elif file_name.endswith((".xlsx", ".xls", ".xlsm")):
            df = pd.read_excel(tmp_path)
            text = df.to_string(index=False)

  
        elif file_name.endswith((".docx", ".doc")):
            doc = Document(tmp_path)
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)

    
        elif file_name.endswith(".txt") or content_type.startswith("text/"):
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        else:
            return f"[Unsupported file type: {file_name}]"

        os.unlink(tmp_path)
        return text.strip()[:5000]  

    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return f"[Extraction failed: {str(e)}]"

class ConversationViewSet(viewsets.ModelViewSet):
    serializer_class = ConversationSerializer
    permission_classes = [IsAuthenticated]
    def create(self, request, *args, **kwargs):
        user = request.user
        since = timezone.now() - timedelta(days=1)
        recent_count = Conversation.objects.filter(user=user, created_at__gte=since).count()

        if recent_count >= MAX_CONVERSATIONS_PER_DAY:
            return Response(
                {"error": f"Daily conversation limit ({MAX_CONVERSATIONS_PER_DAY}) exceeded."},
                status=403
            )
        return super().create(request, *args, **kwargs)

    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user)

    @action(detail=False, methods=['get'], permission_classes=[IsAuthenticated])
    def with_runs(self, request):
        user = request.user
        user_id = request.query_params.get('user_id')
        queryset = Conversation.objects.all()
        if user_id:
            if IsAdmin().has_permission(request, self):
                queryset = queryset.filter(user_id=user_id)
            else:
                return Response(
                    {"error": "You are not allowed to fetch other users' conversations."},
                    status=403
                )
        else:
            queryset = queryset.filter(user=user)
        queryset = queryset.order_by('-created_at')[:10]
        queryset = queryset.prefetch_related('runs__input_files', 'runs__output_artifacts')
        serializer = ConversationWithRunsSerializer(queryset, many=True)
        return Response(serializer.data)


class RegisterView(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    def create(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            token, _ = Token.objects.get_or_create(user=user)
            return Response({
                'message': 'User registered successfully',
                'token': token.key,
                'role': getattr(user, 'role', None)
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginView(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    @action(detail=False, methods=['post'])
    def login(self, request):
        email = request.data.get('email')
        password = request.data.get('password')
        if not email or not password:
            return Response(
                {"error": "Email and password are required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        user = authenticate(request, email=email, password=password)
        if not user:
            return Response(
                {"error": "Invalid credentials"},
                status=status.HTTP_401_UNAUTHORIZED
            )
        token, _ = Token.objects.get_or_create(user=user)
        return Response({
            "token": token.key,
            "id": user.id,
            "email": user.email,
            "role": user.role
        })


class LogoutView(viewsets.ViewSet):
    permission_classes = [permissions.IsAuthenticated]
    @action(detail=False, methods=['post'])
    def logout(self, request):
        try:
            request.user.auth_token.delete()
        except (AttributeError, Token.DoesNotExist):
            pass
        return Response({"message": "User logged out successfully"}, status=status.HTTP_200_OK)


class ReviewViewSet(viewsets.ModelViewSet):
    serializer_class = ReviewSerializer
    permission_classes = [permissions.IsAuthenticated]
    def get_queryset(self):
        user = self.request.user
        if user.role.lower() == "admin":
            return Review.objects.all()
        else:
            return Review.objects.filter(user=user)
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
    def destroy(self, request, *args, **kwargs):
        user = request.user
        if user.role.lower() != "admin":
            return Response({"error": "You do not have permission to delete reviews"}, status=status.HTTP_403_FORBIDDEN)
        return super().destroy(request, *args, **kwargs)


class UserViewSet(viewsets.ModelViewSet):
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]
    def get_queryset(self):
        user = self.request.user
        role = getattr(user, 'role', '').lower()
        if role == 'admin':
            return User.objects.all()
        elif role == 'user':
            return User.objects.filter(id=user.id)
        else:
            return User.objects.none()
    def perform_update(self, serializer):
        user = self.request.user
        role = getattr(user, 'role', '').lower()
        if role == 'admin' or user.id == serializer.instance.id:
            serializer.save()
        else:
            raise PermissionDenied({"error": "You do not have permission to update this user."})
    def perform_destroy(self, instance):
        user = self.request.user
        role = getattr(user, 'role', '').lower()
        if role == 'admin' or user.id == instance.id:
            instance.delete()
        else:
            raise PermissionDenied({"error": "You do not have permission to delete this user."})
    @action(detail=False, methods=['get'], permission_classes=[permissions.IsAuthenticated])
    def me(self, request):
        serializer = self.get_serializer(request.user)
        return Response(serializer.data)


class AgentViewSet(viewsets.ModelViewSet):
    queryset = Agent.objects.all()
    serializer_class = AgentSerializer
    lookup_field = 'agent_id'
    permission_classes = [permissions.IsAuthenticated, IsAdmin]


class ToolViewSet(viewsets.ModelViewSet):
    queryset = Tool.objects.all().order_by('tool_name')
    serializer_class = ToolSerializer
    permission_classes = [permissions.IsAuthenticated, IsAdmin]


class StepViewSet(viewsets.ModelViewSet):
    serializer_class = StepSerializer
    permission_classes = [IsAuthenticated]
    def get_queryset(self):
        user = self.request.user
        if user.role.lower() == "admin":
            return Step.objects.select_related('conversation', 'tool', 'agent').all()
        else:
            return Step.objects.filter(
                conversation__user=user
            ).select_related('conversation', 'tool', 'agent')
    @action(detail=False, methods=['get'])
    def by_conversation(self, request):
        conversation_id = request.query_params.get('conversation_id')
        if not conversation_id:
            return Response({"error": "conversation_id is required"}, status=status.HTTP_400_BAD_REQUEST)
        steps = self.get_queryset().filter(conversation_id=conversation_id).order_by('step_order')
        serializer = self.get_serializer(steps, many=True)
        return Response(serializer.data)


class RunViewSet(viewsets.ModelViewSet):
    queryset = Run.objects.all()
    serializer_class = RunSerializer

    def get_permissions(self):
        if self.action == 'create':
            return [AllowAny()]
        elif self.action in ['list', 'retrieve', 'destroy']:
            return [IsAuthenticated()]
        return [IsAuthenticated()]

    def create(self, request, *args, **kwargs):
        user_input = request.data.get('user_input', '').strip()
        conversation_id = request.data.get('conversation_id', None)

        extracted_texts = []
        uploaded_files = request.FILES.getlist('files')

        for file in uploaded_files:
            text = extract_text_from_file(file)
            if text and not text.startswith("["):
                extracted_texts.append(text)

        file_context = "\n\n--- DOCUMENT BREAK ---\n\n".join(extracted_texts) if extracted_texts else None

        conversation = None
        if conversation_id:
            try:
                conversation = Conversation.objects.get(conversation_id=conversation_id)
                if conversation.user:
                    if not request.user.is_authenticated:
                        return Response({'error': 'Login required to use this conversation'}, status=status.HTTP_403_FORBIDDEN)
                    if conversation.user != request.user:
                        return Response({'error': 'Not allowed to use this conversation'}, status=status.HTTP_403_FORBIDDEN)
            except Conversation.DoesNotExist:
                return Response({'error': 'Conversation not found'}, status=404)

            since = timezone.now() - timedelta(days=1)
            runs_count = Run.objects.filter(conversation=conversation, started_at__gte=since).count()
            if runs_count >= MAX_RUNS_PER_CONVERSATION_PER_DAY:
                return Response(
                    {'error': f'Run limit ({MAX_RUNS_PER_CONVERSATION_PER_DAY}) for this conversation per day exceeded.'},
                    status=403
                )

        run = Run.objects.create(
            user_input=user_input,
            conversation=conversation,
            status=Run.PENDING
        )

        for file in uploaded_files:
            file_type = determine_file_type(file)
            RunInputFile.objects.create(
                run=run,
                file=file,
                file_type=file_type
            )

        if file_context:
            run.final_output = f"__FILE_CONTEXT__:{file_context}"
            run.save(update_fields=['final_output'])

        threading.Thread(target=self.simulate_status, args=(run.id,)).start()

        serializer = RunSerializer(run)
        return Response(serializer.data, status=201)

    def list(self, request, *args, **kwargs):
        user = request.user
        if getattr(user, 'role', '').lower() == 'admin':
            queryset = Run.objects.all()
        else:
            queryset = Run.objects.filter(conversation__user=user)
        serializer = RunSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None, *args, **kwargs):
        try:
            run = Run.objects.get(id=pk)
        except Run.DoesNotExist:
            return Response({'error': 'Run not found'}, status=404)
        if run.conversation and run.conversation.user != request.user:
            return Response({'error': 'Not authorized to view this run'}, status=403)
        serializer = RunSerializer(run)
        return Response(serializer.data)

    def simulate_status(self, run_id):
        try:
            run = Run.objects.get(id=run_id)
            run.status = Run.RUNNING
            run.save(update_fields=['status'])

            file_context = None
            if run.final_output and run.final_output.startswith("__FILE_CONTEXT__:"):
                file_context = run.final_output[len("__FILE_CONTEXT__:"):]
                run.final_output = None
                run.save(update_fields=['final_output'])

            payload = {"query": run.user_input}
            if file_context:
                payload["file_context"] = file_context

            try:
                response = requests.post(
                    ZEN_AGENT_API_URL,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
            except Exception as e:
                run.status = Run.FAILED
                run.final_output = f"Agent request failed: {str(e)}"
                run.save(update_fields=['status', 'final_output'])
                return

            query_type = result.get("type", "rag")
        
            if query_type == "forecast":
                forecast_display = result.get("forecast_display", "Forecast data unavailable")
                interpretation = result.get("interpretation", "No interpretation available")
                confidence_level = result.get("confidence_level", "Medium")
                data_points = result.get("data_points_used", 0)
            
                agent_response = (
                    f"{interpretation}\n\n"
                    f"FORECAST SUMMARY:\n"
                    f"{forecast_display}\n"
                    f"Confidence Level: {confidence_level} ({data_points} data points used)"
                )
            
                dual_forecast = result.get("dual_forecast", {})
                if dual_forecast:
                    RunOutputArtifact.objects.create(
                        run=run,
                        artifact_type="json",
                        data={"dual_forecast": dual_forecast},
                        title="Detailed Forecast Data"
                    )
                
            elif query_type == "scenario":
                agent_response = result.get("llm_analysis", "No scenario analysis available.")
            
            elif query_type == "comparative":
                agent_response = result.get("response", "No comparative analysis available.")
            
            elif query_type == "file_analysis" or query_type == "file_query":
                agent_response = result.get("response", "No response received.")
            
            else:
                agent_response = result.get("response", "No response received.")

            # === NEW: Save artifacts from agent response ===
            artifacts = result.get("artifacts", [])
            for artifact in artifacts:
                RunOutputArtifact.objects.create(
                    run=run,
                    artifact_type="chart",
                    data=artifact,
                    title=artifact.get("title", "Chart")
                )

            graph_url = result.get("graph_url")
            thought_process = result.get("thought_process", [])
            followup = result.get("followup")

            if graph_url:
                RunOutputArtifact.objects.create(
                    run=run,
                    artifact_type="link",
                    data={"url": graph_url},
                    title="Graph Link"
                )
            if thought_process:
                RunOutputArtifact.objects.create(
                    run=run,
                    artifact_type="list",
                    data={"steps": thought_process},
                    title="Thought Process"
                )
            if followup:
                RunOutputArtifact.objects.create(
                    run=run,
                    artifact_type="text",
                    data={"content": followup},
                    title="Follow-up Suggestion"
                )

            run.final_output = agent_response
            run.status = Run.COMPLETED
            run.save(update_fields=['status', 'final_output'])

        except Run.DoesNotExist:
            pass
    def destroy(self, request, *args, **kwargs):
        pk = kwargs.get('pk')
        try:
            run = Run.objects.get(id=pk)
        except Run.DoesNotExist:
            return Response({'error': 'Run not found'}, status=status.HTTP_404_NOT_FOUND)
        user = request.user
        is_admin = getattr(user, 'role', '').lower() == 'admin'
        is_owner = run.conversation and run.conversation.user == user
        if is_admin or is_owner:
            run.delete()
            return Response({'message': 'Run deleted successfully'}, status=status.HTTP_200_OK)
        else:
            return Response({'error': 'You do not have permission to delete this run.'}, status=status.HTTP_403_FORBIDDEN)

class PasswordResetRequestView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = PasswordResetRequestSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            try:
                user = User.objects.get(email=email, is_active=True)
            except User.DoesNotExist:
                return Response({'error': 'User not found.'}, status=status.HTTP_400_BAD_REQUEST)
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            token = default_token_generator.make_token(user)
            reset_link = f"{settings.FRONTEND_URL}/reset/{uid}/{token}/"
            send_mail(
                "Password Reset Requested",
                f"Click the link to reset your password: {reset_link}",
                settings.DEFAULT_FROM_EMAIL,
                [user.email],
                fail_silently=False
            )
            return Response({'message': 'Reset link sent.'}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class PasswordResetConfirmView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = PasswordResetConfirmSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({'message': 'Password reset successful.'}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)