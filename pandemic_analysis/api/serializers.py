from rest_framework import serializers

class PredictionRequestSerializer(serializers.Serializer):
    model = serializers.CharField()
    feature = serializers.CharField()

class FeatureListSerializer(serializers.Serializer):
    features = serializers.ListField(child=serializers.CharField())

class DateRangeSerializer(serializers.Serializer):
    start_date = serializers.DateField()
    end_date = serializers.DateField()
