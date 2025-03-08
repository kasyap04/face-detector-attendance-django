from django.urls import path

from attendence import views

urlpatterns = [
    path("", views.get_attendence, name="attendence"),
    path("check", views.check_attendence, name="Take-Attendence")
]