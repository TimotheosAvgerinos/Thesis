from rest_framework import serializers

class PredictionRequestSerializer(serializers.Serializer):
    model = serializers.CharField()
    feature = serializers.CharField()