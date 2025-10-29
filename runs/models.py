from django.db import models
from conversations.models import Conversation
class Run(models.Model):
    COMPLETED = "COMPLETED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    
    STATUS_CHOICES = [
        (PENDING, 'Pending'),
        (RUNNING, 'Running'),
        (COMPLETED, 'Completed'),
        (FAILED, 'Failed'),
    ]
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name='runs',
        blank=True, null=True
    )
    user_input = models.TextField()
    final_output = models.TextField(blank=True, null=True)
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending'
    )
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Run {self.id} - {self.status} - {self.user_input[:50]}..."


class RunInputFile(models.Model):

    FILE_TYPE_CHOICES = [
        ('pdf', 'PDF Document'),
        ('image', 'Image'),
        ('text', 'Text Summary'),
        ('csv', 'CSV File'),      
        ('excel', 'Excel File'), 
        ('word', 'Word Document'), 
    ]

    run = models.ForeignKey(Run, on_delete=models.CASCADE, related_name='input_files')
    file = models.FileField(upload_to='uploads/')
    file_type = models.CharField(max_length=10, choices=FILE_TYPE_CHOICES)
    description = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return f"{self.file_type.upper()} - {self.file.name} for Run {self.run.id}"

class RunOutputArtifact(models.Model):
    ARTIFACT_TYPE_CHOICES = [
        ('chart', 'Chart (JSON)'),          
        ('table', 'Data Table (JSON)'),     
        ('text', 'Text Summary'),           
        ('pdf_report', 'PDF Report'),      
    ]

    run = models.ForeignKey(Run, on_delete=models.CASCADE, related_name='output_artifacts')
    artifact_type = models.CharField(max_length=20, choices=ARTIFACT_TYPE_CHOICES)
    data = models.JSONField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return f"{self.artifact_type} - {self.title or 'Unnamed'} for Run {self.run.id}"