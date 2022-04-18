from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from api.models import Feedback
from api.utils import predict


@csrf_exempt
def api(request, *args, **kwargs):
    print(request.POST)
    data = request.POST
    text = data["text"]
    target = data["target"]
    mode = data["mode"]
    if mode == "manual":
        modelName = data["modelName"]
        encoding = data["encoding"]
        targetGrouping = data["targetGrouping"]
    if mode == "auto":
        modelName = "auto"
        encoding = "auto"
        targetGrouping = "auto"
    stance, modelName, confidence, result = predict(
        modelName=modelName,
        context=encoding,
        text=text,
        target=target,
        targetGrouping=targetGrouping,
    )
    print(stance, modelName, confidence)

    if mode == "auto":
        # result = [(_, stance, confidence),]
        stance = result[0][1]
        confidence = result[0][2]
        modelName = "cf-n-s-gb"

        # cf-n-s-gb
        # c-a-n-d
        # cf-b-s-gb
        # c-b-n-b
        # cf-n-n-d

        return JsonResponse(
            {
                "mode": mode,
                "text": text,
                "target": target,
                "outputs": [
                    {
                        # "modelName": result[0][0],
                        "modelName": "Context-free Non-GTR Gradient Boosting Classifier",
                        "stance": result[0][1],
                        "confidence": result[0][2],
                    },
                    {
                        # "modelName": result[1][0],
                        "modelName": "Contextual Agglomerative grouping Fully Connected Neural Network",
                        "stance": result[1][1],
                        "confidence": result[1][2],
                    },
                    {
                        # "modelName": result[2][0],
                        "modelName": "Context-free BERTopic grouping Gradient Boosting Classifier",
                        "stance": result[2][1],
                        "confidence": result[2][2],
                    },
                    {
                        # "modelName": result[3][0],
                        "modelName": "Contextual BERTopic grouping Bi-LSTM Neural Network",
                        "stance": result[3][1],
                        "confidence": result[3][2],
                    },
                    {
                        # "modelName": result[4][0],
                        "modelName": "Context-free Non-GTR Fully Connected Neural Network",
                        "stance": result[4][1],
                        "confidence": result[4][2],
                    },
                ],
            }
        )

    if mode == "manual":
        return JsonResponse(
            {
                "stance": stance,
                "text": text,
                "target": target,
                "modelInfo": {
                    "modelName": modelName,
                    "confidence": confidence,
                },
                "mode": mode,
            }
        )


@csrf_exempt
def reportIncorrectPrediction(request, *args, **kwargs):
    print(request.POST)
    data = request.POST
    email = data["email"]
    msg = data["feedback"]
    stance = data["expectedStance"]
    text = data["text"]
    target = data["target"]

    Feedback.objects.create(
        email=email,
        msg=msg,
        stance=stance,
        text=text,
        target=target,
    )
    return HttpResponse()
