from django.urls import path
from .views import ModelListAPIView, PredictAPIView, EvaluationAPIView, PlotAPIView, FeatureListAPIView, DateRangeAPIView

urlpatterns = [
    path('models/', ModelListAPIView.as_view(), name='model-list'),
    path('predict/', PredictAPIView.as_view(), name='predict'),
    path('evaluation/', EvaluationAPIView.as_view(), name='evauation'),
    path('plot/',PlotAPIView.as_view(), name='plot-api'),
    path('features/', FeatureListAPIView.as_view(), name= 'features'),
    path('dates/', DateRangeAPIView.as_view(), name='dates')
]
