from django.db import models
from django.db.models import TextChoices


class AttendenceType(TextChoices):
    FULL_DAY = 'FD', ("FULL_DAY")
    HALF_DAY = 'HD', ("HALF_DAY")




class Student(models.Model):
    id = models.BigAutoField(primary_key=True)  
    name = models.CharField(max_length=225)
    roll_no = models.CharField(max_length=225)
    directory = models.CharField(max_length=225)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'students'



class Attendence(models.Model):
    id = models.BigAutoField(primary_key=True)
    student = models.ForeignKey('Student', models.CASCADE, choices=AttendenceType)
    date = models.DateField()
    attend = models.CharField(max_length=5)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'attendence'
