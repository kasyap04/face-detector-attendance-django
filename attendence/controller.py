import cv2
import os
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

from app.exception import AppException
from app.error import AppError
from attendence.models import Attendence, Student




class AttendenceController:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')



    def train_model(self):
        faces = []
        labels = []

        userlist = os.listdir('static/faces')
        for user in userlist:
            if user == ".gitignore":
                continue

            for imgname in os.listdir(f'static/faces/{user}'):
                img = cv2.imread(f'static/faces/{user}/{imgname}')
                resized_face = cv2.resize(img, (50, 50))
                faces.append(resized_face.ravel())
                labels.append(user)

        faces = np.array(faces)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces, labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')


    def extract_faces(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_points = self.face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
            return face_points
        except:
            return []
        
    def identify_face(self, facearray):
        model = joblib.load('static/face_recognition_model.pkl')
        return model.predict(facearray)
    



    def detect_face(self, frame):
        extract_face = self.extract_faces(frame)
        if len(extract_face) > 0:
            (x, y, w, h) = extract_face[0]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = self.identify_face(face.reshape(1, -1))[0]

            return identified_person
        else:
            raise AppException(*AppError.FACE_NOT_FOUND)
        

    
    def get_attendence(self, date: str):
        if not date:
            raise AppException(*AppError.API_PAYLOAD_NOT_FOUND)
        

        attend = Attendence.objects.filter(date = date).values("student_id")
        data = {
            'attendence': [],
            'students': []
        }

        if attend:
            students = Student.objects.values()
            data['attendence'] = list(map(lambda x: x['student_id'], attend))
            data['students']= list(students)

        return data
    


    def register(self, student_images: dict, student_data):
        name = student_data['name'].strip()
        roll_no = str(student_data['roll_no']).strip()
        print(f'Start with student {name} - {roll_no}')


        if Student.objects.filter(roll_no = roll_no).exists():
             raise AppException(*AppError.STUDENT_ALREADY_REGISTERD)

        face_counts = 0
        stu_dir = f"{name}_{roll_no}"
        folder = f'static/faces/{stu_dir}'

        if not os.path.isdir(folder):
            os.makedirs(folder)


        # img_test = "static/attendence/me.jpg"
        # frame = cv2.imread(img_test)
        # faces = self.extract_faces(frame)

        for index, image in enumerate(student_images.values()):
            frame = cv2.imdecode(
                np.fromstring(image.read(), np.uint8),
                cv2.IMREAD_UNCHANGED
            )
            faces = self.extract_faces(frame)

            if len(faces) > 0:
                face_counts += 1
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(
                    frame, 
                    f'Images Captured: {index}', 
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255, 0, 20), 
                    2, 
                    cv2.LINE_AA
                )
                filename = f"{name}_{index}.jpg"
                print(filename)
                cv2.imwrite(f"{folder}/{filename}", frame[y:y+h, x:x+w])



        if face_counts < 5:
            print(f'Not much face found, removing {folder}')
            os.rmdir(folder)
            raise AppException(*AppError.FACE_NOT_FOUND)
        

        self.train_model()
        
        date = datetime.now()
        stu = Student(
            name=name,
            roll_no=roll_no,
            directory=stu_dir,
            created_at=date,
            updated_at=date
        )
        stu.save()
        print(f"Student {name} - {roll_no} is saved")



    def check_attandance(self, student_images: dict):
        image = student_images['image']

        frame = cv2.imdecode(
            np.fromstring(image.read(), np.uint8),
            cv2.IMREAD_UNCHANGED
        )

        directory = self.detect_face(frame)
        try:
            directory = str(directory)
        except:
            raise AppException(*AppError.FACE_NOT_FOUND)


        student = Student.objects.filter(directory = directory).first()
        if student:
            result = {
                'name': student.name,
                'roll_no': student.roll_no
            }

            date = datetime.now()

            if Attendence.objects.filter(student_id = student.id, date = date).exists():
                result['msg'] = AppError.STUDENT_ALREADY_PRESENT[1]
                return result
                


            att = Attendence(
                student=student,
                date=date.strftime("%Y-%m-%d"),
                attend='FD',
                created_at=date,
                updated_at=date
            )
            att.save()

            return result


        raise AppException(*AppError.FACE_NOT_FOUND)
