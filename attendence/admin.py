from django.contrib import admin
from attendence.models import Student, Attendence


class StudentAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'roll_no', 'directory', 'created_at', 'updated_at')


class AttendenceAdmin(admin.ModelAdmin):
    list_display = ('id', 'date', 'attend', 'created_at', 'updated_at')
 


admin.site.register(Student, StudentAdmin)
admin.site.register(Attendence, AttendenceAdmin)