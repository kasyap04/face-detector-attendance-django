from django.http import JsonResponse

from app.exception import AppException
from attendence.controller import AttendenceController



def get_attendence(request):
    try:
        date = request.POST.get('date')
        controller = AttendenceController()
        result = controller.get_attendence(date)
        return JsonResponse({'status': True, 'data': result})
    except Exception as e:
        print(e)
        return JsonResponse({
            'status': False,
            'code': 500,
            'msg': 'Oopz'
        })




def register_student(request):
    try:
        controller = AttendenceController()
        controller.register(request.FILES, request.POST)
        return JsonResponse(data={'status': True, 'msg': 'Student registered successfully'})
    except AppException as e:
        print(e)
        return JsonResponse(data={
            'status': False,
            'code': e.code,
            'msg': e.msg
        })

    except Exception as e:
        print(e)
        return JsonResponse(status=500, data={
            'status': False,
            'code': 500,
            'msg': 'Oopz'
        })
    


def check_attendence(request):
    try:
        controller = AttendenceController()
        result = controller.check_attandance(request.FILES)
        return JsonResponse(data={'status': True, 'msg': 'Student registered successfully'})
    except AppException as e:
        print(e)
        return JsonResponse(data={
            'status': False,
            'code': e.code,
            'msg': e.msg
        })

    except Exception as e:
        print(e)
        return JsonResponse(status=500, data={
            'status': False,
            'code': 500,
            'msg': 'Oopz'
        })