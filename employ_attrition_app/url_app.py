from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('',views.dashboard,name='dashboard'),
    path('prediction',views.AI_prediction,name='AI_prediction'),
    path('trends',views.Dep_trends,name='Dep_trends'),
    path('employyes',views.employyes,name='employyes'),

#api for insertion
    path('employ_data_insert',views.employ_data_insert,name='employ_data_insert'),

#for testing
    path('sending_response',views.sending_response,name='sending_response')
    
]