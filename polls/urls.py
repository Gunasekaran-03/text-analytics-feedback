from django.urls import path

from . import views

urlpatterns = [
    path('', views.process_df, name='result'),

    path('file-upload/', views.process_df, name='result'),
    path('Dashboard', views.index, name='index'),
    path('download-df/', views.downloadDf, name='download'),
    path('download-full-df/', views.downloadfullDf, name='downloadfulldf'),
    path('dashboard/', views.search_df, name='search'),
    path('current-dataset/', views.dataset_view, name='viewdf'),
    path('full-dataset/', views.full_dataset_view, name='fullviewdf')

]