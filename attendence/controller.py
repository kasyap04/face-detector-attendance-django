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



    # def train_model(self):
    #     faces = []
    #     labels = []

    #     userlist = os.listdir('static/faces')
    #     for user in userlist:
    #         if user == ".gitignore":
    #             continue

    #         for imgname in os.listdir(f'static/faces/{user}'):
    #             img = cv2.imread(f'static/faces/{user}/{imgname}')
    #             resized_face = cv2.resize(img, (50, 50))
    #             faces.append(resized_face.ravel())
    #             labels.append(user)

    #     faces = np.array(faces)
    #     knn = KNeighborsClassifier(n_neighbors=5)
    #     knn.fit(faces, labels)
    #     joblib.dump(knn, 'static/face_recognition_model.pkl')


    def train_model(self):
        faces = []
        labels = []

        userlist = os.listdir('static/faces')
        for user in userlist:
            if user == ".gitignore":
                continue

            for imgname in os.listdir(f'static/faces/{user}'):
                img = cv2.imread(f'static/faces/{user}/{imgname}')
                
                # Ensure the image was loaded properly
                if img is None:
                    print(f"Warning: Could not load image {imgname} for user {user}")
                    continue
                    
                resized_face = cv2.resize(img, (50, 50))
                
                # Flatten the image to 1D array
                flattened_face = resized_face.ravel()
                
                faces.append(flattened_face)
                labels.append(user)
        print(faces,labels)
        faces = np.array(faces)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces, labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')
        # faces = []
        # labels = []

        # userlist = os.listdir('static/faces')
        # for user in userlist:
        #     if user == ".gitignore":
        #         continue

        #     for imgname in os.listdir(f'static/faces/{user}'):
        #         img = cv2.imread(f'static/faces/{user}/{imgname}')
                
        #         # Use a face detector to extract only the face region (important!)
        #         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #         detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
        #         if len(detected_faces) > 0:
        #             # Take the largest face
        #             (x, y, w, h) = max(detected_faces, key=lambda rect: rect[2] * rect[3])
        #             face_img = img[y:y+h, x:x+w]
        #             resized_face = cv2.resize(face_img, (50, 50))
                    
        #             # Normalize the image
        #             normalized_face = resized_face.astype('float') / 255.0
                    
        #             faces.append(normalized_face.ravel())
        #             labels.append(user)

        # faces = np.array(faces)
        
        # # Use a more sophisticated distance metric
        # knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
        # knn.fit(faces, labels)
        
        # # Also save the threshold information
        # # Calculate average distances between known faces
        # threshold_data = {}
        # threshold_data['threshold'] = 0.5  # Default threshold
        
        # joblib.dump(knn, 'static/face_recognition_model.pkl')
        # joblib.dump(threshold_data, 'static/threshold_data.pkl')


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

    # def identify_face(self, facearray):
    #     model = joblib.load('static/face_recognition_model.pkl')    
        
    #     if len(facearray.shape) == 3:
    #         flattened_face = facearray.ravel()
    #     else:
    #         flattened_face = facearray
        

    #     flattened_face = flattened_face.reshape(1, -1)
    #     distances, indices = model.kneighbors(flattened_face, return_distance=True)
    #     distances = distances[0]

    #     threshold = 13873 
        
    #     if min(distances) > threshold:
    #         return ["unknown"]
    #     else:
    #         return model.predict(flattened_face)


    def detect_face(self, frame):
        extract_face = self.extract_faces(frame)
        # self.detect_multiple_face(frame)
        if len(extract_face) > 0:
            (x, y, w, h) = extract_face[0]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = self.identify_face(face.reshape(1, -1))[0]

            return identified_person
        else:
            raise AppException(*AppError.FACE_NOT_FOUND)
        
    def detect_multiple_face(self, frame):
        # path = 'static/test'
        # userlist = os.listdir(path)
        # for img in userlist:
            # frame = cv2.imread(f"{path}/{img}")

        # frame = cv2.imread(f"static/faces/Vishnu_25/Vishnu_2.jpg")
        # frame = cv2.imread(f"static/test/qw_1.jpg")


        
        extract_face = self.extract_faces(frame)
        for index, f in enumerate(extract_face):
            # (x, y, w, h) = extract_face[0]
            (x, y, w, h) = f


            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = self.identify_face(face.reshape(1, -1))[0]

            print(f"{identified_person = }")

            yield identified_person

            # cv2.imwrite(f'static/test/{identified_person}_{index}.jpg', frame[y:y+h, x:x+w])

        # cv2.imwrite(f'static/test/jin_res.jpeg', frame[y:y+h, x:x+w])



        

    
    def get_attendence(self, date: str):
        if not date:
            raise AppException(*AppError.API_PAYLOAD_NOT_FOUND)
        

        attend = Attendence.objects.filter(date = date).values("student_id")
        data = {
            'attendence': [],
            'students': []
        }

        if attend:
            students = Student.objects.order_by('roll_no').values()
            data['attendence'] = list(map(lambda x: x['student_id'], attend))
            data['students']= list(students)

        return data
    


    def register(self, student_images: dict, student_data):
        name = student_data['name'].strip()
        roll_no = str(student_data['roll_no']).strip()
        print(f'Start with student {name} - {roll_no}')


        for index, image in enumerate(student_images.values()):
            frame = cv2.imdecode(
                    np.fromstring(image.read(), np.uint8),
                    cv2.IMREAD_UNCHANGED
                )
            for zind, face in enumerate(self.extract_faces(frame)):
                (x, y, w, h) = face
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
                filename = f"{name}_{zind}.jpg"
                print(filename)
                cv2.imwrite(f'static/test/{filename}', frame[y:y+h, x:x+w])


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


        for d in self.detect_multiple_face(frame):

            try:
                directory = str(d)
            except:
                raise AppException(*AppError.FACE_NOT_FOUND)
            else:
                directory = d

            student = Student.objects.filter(directory = directory).first()
            if student:
                result = {
                    'msg': "Attendence marked"
                }

                date = datetime.now()

                if Attendence.objects.filter(student_id = student.id, date = date).exists():
                    continue
                    


                att = Attendence(
                    student=student,
                    date=date.strftime("%Y-%m-%d"),
                    attend='FD',
                    created_at=date,
                    updated_at=date
                )
                att.save()

        return {
            'msg': "Attendence marked"
        }
