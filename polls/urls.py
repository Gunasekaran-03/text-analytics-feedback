from django.urls import path

from . import views

urlpatterns = [
    path('dashboard/', views.process_df, name='result'),
    path('', views.index, name='index'),
    path('download/', views.downloadDf, name='download'),
    path('Dashboard/', views.search_df, name='search'),
    path('current-dataset/', views.dataset_view, name='viewdf'),
    path('full-dataset/', views.full_dataset_view, name='fullviewdf')

]