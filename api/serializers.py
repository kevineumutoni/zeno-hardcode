from rest_framework import serializers
from users.models import User, Review
from conversations.models import Conversation
from agents.models import Agent, Tool
from runs.models import RunInputFile, RunOutputArtifact, Run
from conversations.models import Step
from rest_framework.exceptions import ValidationError
from django.utils.encoding import force_str
from django.utils.http import urlsafe_base64_decode
from django.contrib.auth.tokens import default_token_generator
import re


class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True)

    class Meta:
        model = User
        fields = ['id', 'first_name', 'last_name', 'email', 'role', 'image', 'created_at', 'password']
        extra_kwargs = {
            'first_name': {'required': True, 'allow_blank': False},
            'last_name': {'required': True, 'allow_blank': False},
            'email': {'required': True, 'allow_blank': False},
        }

    def validate_password(self, value):
        if len(value) < 8:
            raise ValidationError("Password must be at least 8 characters long.")
        if not re.search(r'[A-Z]', value):
            raise ValidationError("Password must contain at least one uppercase letter.")
        if not re.search(r'\d', value):
            raise ValidationError("Password must contain at least one number.")
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', value):
            raise ValidationError("Password must contain at least one special character.")
        return value

    def create(self, validated_data):
        password = validated_data.pop('password')
        user = User.objects.create_user(**validated_data, password=password)
        return user


class PasswordResetRequestSerializer(serializers.Serializer):
    email = serializers.EmailField()

    def validate_email(self, value):
        if not User.objects.filter(email=value, is_active=True).exists():
            raise ValidationError("Email not found or inactive.")
        return value


class PasswordResetConfirmSerializer(serializers.Serializer):
    uidb64 = serializers.CharField()
    token = serializers.CharField()
    new_password = serializers.CharField(write_only=True)
    confirm_password = serializers.CharField(write_only=True)

    def validate(self, data):
        if data['new_password'] != data['confirm_password']:
            raise ValidationError("Passwords do not match.")
        password = data['new_password']
        if len(password) < 6:
            raise ValidationError("Password must be at least 6 characters long.")
        if not re.search(r'[A-Z]', password):
            raise ValidationError("Password must contain at least one uppercase letter.")
        if not re.search(r'\d', password):
            raise ValidationError("Password must contain at least one number.")
        if not re.search(r'[!@#$%^&*(),.?\":{}|<>]', password):
            raise ValidationError("Password must contain at least one special character.")

        try:
            uid = force_str(urlsafe_base64_decode(data['uidb64']))
            user = User.objects.get(pk=uid)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            raise ValidationError("Invalid user.")
        if not default_token_generator.check_token(user, data['token']):
            raise ValidationError("Invalid or expired token.")
        data['user'] = user
        return data

    def save(self):
        user = self.validated_data['user']
        user.set_password(self.validated_data['new_password'])
        user.save()
        return user


class ConversationSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    user_id = serializers.PrimaryKeyRelatedField(
        queryset=User.objects.all(),
        source='user',
        write_only=True
    )

    class Meta:
        model = Conversation
        fields = ['conversation_id', 'user', 'user_id', 'title', 'created_at']
        read_only_fields = ['conversation_id', 'created_at']
        extra_kwargs = {
            'title': {'required': False}
        }

    def create(self, validated_data):
        if 'title' not in validated_data or not validated_data['title']:
            validated_data['title'] = "New Chat"
        return super().create(validated_data)


class ReviewSerializer(serializers.ModelSerializer):
    user = serializers.PrimaryKeyRelatedField(read_only=True)

    class Meta:
        model = Review
        fields = ['review_id', 'review_text', 'rating', 'created_at', 'user']


class AgentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Agent
        fields = ['agent_id', 'agent_name', 'description']


class ToolSerializer(serializers.ModelSerializer):
    tool_name = serializers.CharField(required=True, allow_blank=False)
    tool_description = serializers.CharField(required=True, allow_blank=False)
    meta_data = serializers.JSONField(required=True)

    class Meta:
        model = Tool
        fields = '__all__'


class RunInputFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = RunInputFile
        fields = ['id', 'file', 'file_type', 'description']
        read_only_fields = ['id']


class RunOutputArtifactSerializer(serializers.ModelSerializer):
    """
    Handles chart/graph artifacts from agent (Ethiopia/Kenya forecasts).
    Example `data`:
        {
            "x": ["2025", "2026", "2027"],
            "y": [380, 320, 300],
            "title": "Price Forecast",
            "chart_type": "line",
            "y_axis_label": "USD"
        }
    """
    data = serializers.JSONField()  # Accepts any dict → perfect for charts
    artifact_type = serializers.CharField(
        default="chart",
        read_only=True
    )

    class Meta:
        model = RunOutputArtifact
        fields = ['id', 'artifact_type', 'data', 'title']
        read_only_fields = ['id', 'artifact_type']


class RunSerializer(serializers.ModelSerializer):
    input_files = RunInputFileSerializer(many=True, read_only=True)
    output_artifacts = RunOutputArtifactSerializer(many=True, read_only=True)
    conversation_id = serializers.PrimaryKeyRelatedField(
        source="conversation",
        queryset=Conversation.objects.all(),
        required=False,
        allow_null=True
    )

    class Meta:
        model = Run
        fields = '__all__'
        read_only_fields = [
            'id', 'status', 'final_output',
            'input_files', 'output_artifacts'
        ]


class StepSerializer(serializers.ModelSerializer):
    conversation = serializers.PrimaryKeyRelatedField(
        queryset=Conversation.objects.all()
    )
    tool = serializers.PrimaryKeyRelatedField(
        queryset=Tool.objects.all(),
        required=False,
        allow_null=True
    )
    agent = serializers.PrimaryKeyRelatedField(
        queryset=Agent.objects.all(),
        required=False,
        allow_null=True
    )

    class Meta:
        model = Step
        fields = [
            'step_id',
            'conversation',
            'step_order',
            'type',
            'content',
            'tool',
            'agent',
            'created_at'
        ]
        read_only_fields = ['step_id', 'created_at']

    def validate(self, data):
        step_type = data.get('type')
        tool = data.get('tool')
        agent = data.get('agent')

        if step_type == 'tool_call' and not tool:
            raise serializers.ValidationError({
                'tool': "This field is required when type is 'tool_call'."
            })
        if step_type == 'sub_agent_call' and not agent:
            raise serializers.ValidationError({
                'agent': "This field is required when type is 'sub_agent_call'."
            })

        if step_type != 'tool_call' and tool:
            raise serializers.ValidationError({
                'tool': "This field should only be set when type is 'tool_call'."
            })
        if step_type != 'sub_agent_call' and agent:
            raise serializers.ValidationError({
                'agent': "This field should only be set when type is 'sub_agent_call'."
            })

        return data


class ConversationWithRunsSerializer(serializers.ModelSerializer):
    runs = RunSerializer(many=True, read_only=True)

    class Meta:
        model = Conversation
        fields = ['conversation_id', 'title', 'created_at', 'runs']