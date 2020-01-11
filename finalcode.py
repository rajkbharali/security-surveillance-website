from flask import Flask, render_template, Response,session,  request, redirect, g, url_for, flash, stream_with_context
import cv2
import numpy as np
import imutils
import pandas as pd
import mysql.connector
import os
import time
import face_recognition
import json
import shutil
import datetime
import threading
from werkzeug.utils import secure_filename
from werkzeug.wrappers import Request, Response
import cv2
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException, NotFound
import bcrypt
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import psutil

#sconnecting database..

mydb = mysql.connector.connect(
  host="localhost",
  user="project",
  passwd="12qwaszx",
  database="ip_webcam"
)


#fetching datas from tabels
db_cursor = mydb.cursor()
db_cursor.execute('SELECT * FROM camdetail')
table_rows = db_cursor.fetchall()
df = pd.DataFrame(table_rows)

#displaying safe bank
db_cursor.execute('SELECT * FROM safe')
table_rows = db_cursor.fetchall()
df1 = pd.DataFrame(table_rows)

#displaying unsafe bank
db_cursor.execute('SELECT * FROM unsafe')
table_rows = db_cursor.fetchall()
df2 = pd.DataFrame(table_rows)

#displaying unknown bank
db_cursor.execute('SELECT * FROM unknown')
table_rows = db_cursor.fetchall()
df4 = pd.DataFrame(table_rows)


#displaying advanced settings
db_cursor.execute('SELECT * FROM settings')
table_rows = db_cursor.fetchall()
df3 = pd.DataFrame(table_rows)

# to'pandas.core.frame.DataFrame'

#convertings all the data into list
p=df.values.tolist()
list1=df1.values.tolist()
list2=df2.values.tolist()
list3=df3.values.tolist()
list4=df4.values.tolist()


count=0
person = 0
total_frames = 0

lock = threading.Lock()


#defining app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "/home/ubuntu/opencv/static/threat"
app.config['UPLOAD_FOLDER1'] = "/home/ubuntu/opencv/static/safe"
app.config['UPLOAD_FOLDER2'] = "/home/ubuntu/opencv/static/unknown"
app.secret_key = os.urandom(24)


if len(p)!=0:
        @app.route('/')
        def index():
                """ Video streaming home page """
                global rows,column,db_cursor,p,list1,list2,list3,list4,df,df1,df2,df3,df4,age,gend,col
                db_cursor = mydb.cursor()
                db_cursor.execute('SELECT * FROM camdetail')
                table_rows = db_cursor.fetchall()
                df = pd.DataFrame(table_rows)
                
                db_cursor.execute('SELECT * FROM safe')
                table_rows = db_cursor.fetchall()
                df1 = pd.DataFrame(table_rows)
                
                db_cursor.execute('SELECT * FROM unsafe')
                table_rows = db_cursor.fetchall()
                df2 = pd.DataFrame(table_rows)
                
                #displaying unknown bank
                db_cursor.execute('SELECT * FROM unknown')
                table_rows =db_cursor.fetchall()
                df4 = pd.DataFrame(table_rows)
                
                db_cursor.execute('SELECT * FROM settings')
                table_rows = db_cursor.fetchall()
                df3 = pd.DataFrame(table_rows)
                # to'pandas.core.frame.DataFrame'
                
                p=df.values.tolist()
                list1=df1.values.tolist()
                list2=df2.values.tolist()
                list3=df3.values.tolist()
                list4=df4.values.tolist()
                
                x=list3[0][9]
                y=list3[0][10]
                age=list3[0][11]
                gend=list3[0][12]
                col=list3[0][13]
                return render_template('p.html',url=len(p),rows=int(x),column=int(y))
        def detectmotion(url1):

            global x,name, frame,im,count,face,font,age,gend,col,person,total_frames
            im=[]
            Labels = []
            
            if os.path.exists("yolo-coco/9k.names"):
                print("")
                with open("yolo-coco/coco.names") as k:
                    Labels = k.read().strip().split("\n")
            else:
                print("no")
            np.random.seed(42)
            COLORS = np.random.randint(0, 255, size=(len(Labels), 3), dtype="uint8")

            weightsPath = "/home/ubuntu/opencv/yolo-coco/yolov3.weights"
            configPath = "/home/ubuntu/opencv/yolo-coco/yolov3.cfg"

            #print("[INFO] loading YOLO from disk...")
            net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

            video = WebcamVideoStream(src=p[url1][2]).start()
            fps = FPS().start()

            known_face_encodings = []
            known_face_names =[]

            for f in list1:
                    x = 'safe'+f[1]
                    known_face_names.append(x)
                    root = os.path.join('.','static/safe',f[2])
                    li= face_recognition.load_image_file(root)
                    fe = face_recognition.face_encodings(li)
                    known_face_encodings.extend(fe)


            for f in list2:
                    x='danger'+f[1]
                    known_face_names.append(x)
                    root = os.path.join('.','static/threat',f[2])
                    li=face_recognition.load_image_file(root)
                    fe = face_recognition.face_encodings(li)
                    known_face_encodings.extend(fe)

            font = cv2.FONT_HERSHEY_SIMPLEX
            MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
            age_list = ['(0,2)','(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
            gender_list = ['Male', 'Female']
           
            age_net = cv2.dnn.readNetFromCaffe(
                'data/deploy_age.prototxt',
                'data/age_net.caffemodel')

            gender_net = cv2.dnn.readNetFromCaffe(
                'data/deploy_gender.prototxt',
                'data/gender_net.caffemodel')
   
            while fps._numFrames <100:
                    with lock:
                        #print(threading.current_thread().name,url1)
                        #process = psutil.Process(os.getpid())
                        #print(process.memory_percent())
                    
                        frame = video.read()
                    
                        frame = imutils.resize(frame,width=500)
                        (H, W) = frame.shape[:2]
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # object detection using yolo 
                        ln = net.getLayerNames()
                        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
                        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                        net.setInput(blob)
                        layerOutputs = net.forward(ln)

                        boxes = []
                        confidences = []
                        classIDs = []

                        for output in layerOutputs:
                            for detection in output:
                                scores = detection[5:]
                                classID = np.argmax(scores)
                                confidence = scores[classID]
                                if confidence > 0.5:
                                    box = detection[0:4] * np.array([W, H, W, H])
                                    (centerX, centerY, width, height) = box.astype("int")
                                    x = int(centerX - (width / 2))
                                    y = int(centerY - (height / 2))
                                    boxes.append([x, y, int(width), int(height)])
                                    confidences.append(float(confidence))
                                    classIDs.append(classID)

                        persons=[]
                        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.3)

                        if len(idxs) > 0:
                            for i in idxs.flatten():
                                (x, y) = (boxes[i][0], boxes[i][1])
                                (w, h) = (boxes[i][2], boxes[i][3])
                                
                                color = [int(c) for c in COLORS[classIDs[i]]]
                                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                texts = "{}: {:.4f}".format(Labels[classIDs[i]], confidences[i])
                                cv2.putText(frame, texts, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                                if Labels[classIDs[i]]=="person":
                                    persons.append(Labels[classIDs[i]])


                        timestamp = datetime.datetime.now()
                        cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(frame, p[url1][3],(10, frame.shape[0] - 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(frame,p[url1][1],(400, frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        rgb_frame = frame[:, :, ::-1]

                        face_locations = face_recognition.face_locations(rgb_frame)
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                        gender1=[]
                        age1=[]
                        face_names = []
                        face_names1 = []
                        
                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                            name = "Unknown"
                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = known_face_names[best_match_index]

                            face_img = frame
                            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                            
                            gender_net.setInput(blob)
                            gender_preds = gender_net.forward()
                            gender = gender_list[gender_preds[0].argmax()]
                            gender1.append(gender)

                            age_net.setInput(blob)
                            age_preds = age_net.forward()
                            age = age_list[age_preds[0].argmax()]
                            age1.append(age)
                            
                            count +=1
                            count1 = count + int(timestamp.strftime("%m%d%I%M%S"))
                            if name=="Unknown" :
                                    face_names1.append(name)
                                    cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
                                    y = top - 15 if top - 15 > 15 else top + 15
                                    cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                                    c=cv2.imwrite('/home/ubuntu/opencv/static/unknown/frame%d.jpg'%count1,frame )
                                    print(c)
                                    x='frame'+str(count1)+'.jpg'
                                    sqlformula="INSERT INTO unknown(id,image) VALUES (%s,%s)"
                                    room1=(count1,x)
                                    db_cursor.execute(sqlformula,room1)
                                    mydb.commit()
                                    
                            elif name[:6] =="danger":
                                    face_names.append(name)
                                    cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255),2)
                                    # Draw a label with a name below the face
                                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), cv2.FILLED)
                                    font = cv2.FONT_HERSHEY_DUPLEX
                                    cv2.putText(frame, name[6:], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                                    

                            elif name[:4]== "safe":
                                    face_names.append(name)
                                    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0),3)
                                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,225,0), cv2.FILLED)
                                    font = cv2.FONT_HERSHEY_DUPLEX
                                    cv2.putText(frame, name[4:], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                            temp=len(persons)-len(face_names)
                            x1 = str(gender1)
                            x2 = str(age1)
                            x3 = len(persons)
                            x4 = str(timestamp.strftime("%Y-%m-%d"))
                            x5 = p[url1][3]
                            x6 = p[url1][1]
                            x7=str(timestamp.strftime("%H:%M:%S"))
                            known = len(face_names)
                            unknown = str(temp)
                            sqlformula="INSERT INTO info(id,camera,location,date,time,total_person,gender,age,known,unknown) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                            room1=(count1,x5,x6,x4,x7,x3,x1,x2,known,unknown)
                            db_cursor.execute(sqlformula,room1)
                            mydb.commit()

                            for i,j in zip(age1,gender1):
                                    y = str(i)
                                    y1= str(j)
                                    sqlformula="INSERT INTO age_gender(frame_id,age,gender) VALUES (%s,%s,%s)"
                                    room=(count1,y,y1)
                                    db_cursor.execute(sqlformula,room)
                                    mydb.commit()

                        
                                      
                    if 1>0:
                            frame1 = cv2.imencode(".jpg", frame)[1].tobytes()
                            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')
                            #open('t.jpg','rb').read()
                            key = cv2.waitKey(1) & 0xFF    

        
        @app.route('/login1',methods=['GET','POST'])
        def login1():
            
            session.permanent = True
            app.permanent_session_lifetime = timedelta(minutes=1)
            
            if 'username' in session:
                username = session['username']
                p=request.form['password'].encode('utf-8')
                
                if bcrypt.hashpw(p,user["password"].encode('utf-8')) == user["password"].encode('utf-8'):
                    return 'Logged in as ' + username + '<br>' + \
                     "<b><a href = '/logout'>click here to log out</a></b>"
                
            return "You are not logged in <br><a href = '/login'></b>" + \
                "click here to log in</b></a>"
        
        ##################################################################################################################
        
        #login page
        
        @app.route('/login', methods = ['GET', 'POST'])
        def login():
            if request.method == "POST":
                username = request.form['username']
                password = request.form['password'].encode('utf-8')
                cur = mydb.cursor(dictionary=True)
                cur.execute("SELECT * FROM users WHERE username=%s",(username,))
                user = cur.fetchone()
                cur.close()
                
                if len(user) > 0:
                    if bcrypt.hashpw(password,user["password"].encode('utf-8')) == user["password"].encode('utf-8'):
                    #Create a user login session
                        session['name'] = user['name']
                        session['username'] = user['username']
                        return redirect(url_for("chart"))
                    else:
                        return "Enter Password  Or Username Not Match!"
                else:
                    return "Enter Password  Or Username Not Match!"
            else:
                return render_template('login.html')

        ##################################################################################################################
        #Register Page

        @app.route('/register',methods=["POST","GET"])
        def register():
            if request.method=="GET":
                redirect('login')
            else:
                name = request.form['name']
                email = request.form['email']
                username = request.form['username']
                password = request.form['password'].encode('utf-8')
                hash_password = bcrypt.hashpw(password,bcrypt.gensalt())
                cur=mydb.cursor()
                cur.execute("INSERT INTO users(name,username,email,password) VALUES(%s,%s,%s,%s)",
                    (name,username,email,hash_password,))
                mydb.commit()
                return redirect(url_for("login"))
            return render_template("register.html")
############################################################################################
        @app.errorhandler(404)
        def page_not_found(e):
            return redirect('/dashboard'),404
#################################################################################################################
        @app.route('/dashboard', methods=['GET','POST'])
        def chart():
          if 'username' in session:
              username = session['username']
              if request.method == "POST":
                  start=request.form['start']
                  end=request.form['end']
                  cam = request.form.get('drop1')
                  db_cursor = mydb.cursor()
                  db_cursor.execute("SELECT sum(known),sum(unknown),sum(total_person),date FROM ip_webcam.info WHERE date BETWEEN %s AND %s And camera=%s GROUP BY date",(start,end,cam))
                  data = db_cursor.fetchall()
                  known=[]
                  unknown=[]
                  total=[]
                  labels=[]
                  for i in range(len(data)):
                    known.append(data[i][0])
                  for i in range(len(data)):
                    unknown.append(data[i][1])
                  for i in range(len(data)):
                    total.append(data[i][2])
                  for i in range(len(data)):
                    labels.append(data[i][3]) 
                  a={'total':total,'known':known,'unknown':unknown, 'labels':labels}
                  db_cursor.execute("SELECT  camera_info From ip_webcam.camdetail")
                  camera=db_cursor.fetchall()
                  camp=[]
                  for i in range(len(camera)):
                    camp.append(camera[i][0])
                  camp=set(camp)

                  #################Line Chart##################################
                  lstart =request.form['start']
                  lend =request.form['end']
                  lcam = request.form.get('drop2')
                  db_cursor.execute("select info.camera,sum(total_person),sum(known),sum(unknown),count(case when age_gender.gender='Male' then 1 end) AS Male,count(case when age_gender.gender='Female' then 1 end) AS Female,info.date from info inner join age_gender where date BETWEEN %s AND %s And info.camera=%s And info.id=age_gender.frame_id group by date",(lstart,lend,lcam))
                  data=db_cursor.fetchall()
                  camera=[]
                  person=[]
                  known=[]
                  unknown=[]
                  male=[]
                  female=[]
                  date=[]
                  for i in range(len(data)):
                    camera.append(data[i][0])
                  for i in range(len(data)):
                    person.append(data[i][1])
                  for i in range(len(data)):
                    known.append(data[i][2])
                  for i in range(len(data)):
                    unknown.append(data[i][3])
                  for i in range(len(data)):
                    male.append(data[i][4])
                  for i in range(len(data)):
                    female.append(data[i][5])
                  for i in range(len(data)):
                    date.append(data[i][6])  
                  data={'camera':camera,'person':person,'known':known,'unknown':unknown,'male':male,'female':female,'date':date}
                  ##############33#Pie Chart############################################
                  plocation = request.form.get('loc')
                  pcamera = request.form.get('cam')
                  db_cursor.execute("SELECT info.camera,info.location,count(case when age_gender.gender='Male' then 1 end) AS Male,count(case when age_gender.gender='Female' then 1 end) AS Female,info.date from info join age_gender on info.id=age_gender.frame_id WHERE info.location=%s And info.camera=%s",(plocation,pcamera))
                  pdata = db_cursor.fetchall()
                  camera=[]
                  location=[]
                  male=[]
                  female=[]
                  date=[]
                  for i in range(len(pdata)):
                        camera.append(pdata[i][0])
                  for i in range(len(pdata)):
                        location.append(pdata[i][1])
                  for i in range(len(pdata)):
                        male.append(pdata[i][2])
                  for i in range(len(pdata)):
                        female.append(pdata[i][3])
                  for i in range(len(pdata)):
                        date.append(pdata[i][4])
                  b={'camera':camera,'location':location,'male':male, 'female':female,'date':date}
                  plocation=set(location)
                  loc = list(plocation)
                  ##################Age Pie Chart####################################
                  aplocation = request.form.get('loc')
                  apcamera = request.form.get('cam')
                  db_cursor.execute("SELECT count(age_gender.age) AS age_count,age_gender.age AS age from info join age_gender on info.id=age_gender.frame_id WHERE info.location=%s And info.camera=%s Group By age",(aplocation,apcamera))
                  apdata = db_cursor.fetchall()
                  age_count=[]
                  age_range=[]
                  for i in range(len(data)):
                        age_count.append(data[i][0])
                  for i in range(len(data)):
                        age_range.append(data[i][1])
                  c={'age_count':age_count,'age_range':age_range}
                  return render_template('graph.html',a=a,data=data,camera=camp,b=b,location=loc,c=c)       
              else:
                  db_cursor = mydb.cursor()
                  db_cursor.execute("SELECT sum(known),sum(unknown),sum(total_person),camera,date FROM ip_webcam.info GROUP BY date")
                  data = db_cursor.fetchall()
                  known=[]
                  unknown=[]
                  total=[]
                  labels=[]
                  cam=[]
                  for i in range(len(data)):
                    known.append(data[i][0])
                  for i in range(len(data)):
                    unknown.append(data[i][1])
                  for i in range(len(data)):
                    total.append(data[i][2])
                  for i in range(len(data)):
                    cam.append(data[i][3])
                  for i in range(len(data)):
                    labels.append(data[i][4]) 
	
                  a={'total':total,'known':known,'unknown':unknown, 'labels':labels}
                  #Line Chart
                  db_cursor.execute("select info.camera,sum(total_person),sum(known),sum(unknown) from info  group by date")
                  data=db_cursor.fetchall()
                  db_cursor.execute("select count(case when age_gender.gender='Male' then 1 end) AS Male,count(case when age_gender.gender='Female' then 1 end) AS Female,info.date from info inner join age_gender where  info.id=age_gender.frame_id group by date")
                  dt=db_cursor.fetchall()                  
                  camera=[]
                  person=[]
                  known=[]
                  unknown=[]
                  male=[]
                  female=[]
                  date=[]
                  for i in range(len(data)):
                    camera.append(data[i][0])
                  for i in range(len(data)):
                    person.append(data[i][1])
                  for i in range(len(data)):
                    known.append(data[i][2])
                  for i in range(len(data)):
                    unknown.append(data[i][3])
                  for i in range(len(dt)):
                    male.append(dt[i][0])
                  for i in range(len(dt)):
                    female.append(dt[i][1])
                  for i in range(len(dt)):
                    date.append(dt[i][2])  
                  data={'camera':camera,'person':person,'known':known,'unknown':unknown,'male':male,'female':female,'date':date}
                  cam=set(cam)
                  #################################Gender Pie Chart#############################
                  db_cursor.execute("select info.camera,info.location,count(case when age_gender.gender='Male' then 1 end) AS Male,count(case when age_gender.gender='Female' then 1 end) AS Female,info.date from info join age_gender on info.id=age_gender.frame_id")
                  pdata = db_cursor.fetchall()
                  camera=[]
                  location=[]
                  male=[]
                  female=[]
                  date=[]
                  for i in range(len(pdata)):
                        camera.append(pdata[i][0])
                  for i in range(len(pdata)):
                        location.append(pdata[i][1])
                  for i in range(len(pdata)):
                        male.append(pdata[i][2])
                  for i in range(len(pdata)):
                        female.append(pdata[i][3])
                  for i in range(len(pdata)):
                        date.append(pdata[i][4])
                  ploc=[]
                  pcam=[]
                  db_cursor.execute("select Location,camera_info From  ip_webcam.camdetail")
                  pdata = db_cursor.fetchall()
                  for i in range(len(pdata)):
                    ploc.append(pdata[i][0])
                  for i in range(len(pdata)):
                    pcam.append(pdata[i][1])
                  ploc=set(ploc)
                  pcam=set(pcam)
                  b={'camera':camera,'location':ploc,'male':male, 'female':female,'date':date}
                  #######################Age Pie Chart######################################
                  db_cursor.execute("SELECT count(age_gender.age) AS age_count,age_gender.age AS age from info join age_gender on info.id=age_gender.frame_id Group By age")
                  apdata = db_cursor.fetchall()
                  age_count=[]
                  age_range=[]
                  for i in range(len(apdata)):
                        age_count.append(apdata[i][0])
                  for i in range(len(apdata)):
                        age_range.append(apdata[i][1])
                  c={'age_count':age_count, 'age_range':age_range}
                  return render_template('graph.html',a=a,data=data,camera=cam,b=b,ploc=ploc,pcam=pcam,c=c)
          return "You are not logged in <br><a href = '/login'></b>" + \
            "click here to log in</b></a>"


                    
#################################################################################################################
        # Displaying Safe Bank Detail
        
        @app.route('/safelist', methods=['GET', 'POST'])
        def safelist():
            
            if 'username' in session:
                db_cursor.execute('SELECT * FROM safe')
                table_rows = db_cursor.fetchall()
                df1 = pd.DataFrame(table_rows)
                list1=df1.values.tolist()
                print("lisss",table_rows)
                id=[]
                name=[]
                image=[]
                for i in range(len(table_rows)):
                    id.append(table_rows[i][0])
                    name.append(table_rows[i][1])
                    image.append(table_rows[i][2])
                lis={'id':id,'name':name,'image':image}
                return render_template('safedisp.html',tab=table_rows,ta=lis,len=len(table_rows))  
            return "You are not logged in <br><a href = '/login'></b>" + \
                "click here to log in</b></a>"
            
            
        ##########################################################
        
        
        # Inserting/Adding Form in Safe Bank 
        
        @app.route('/safe',methods=['GET','POST'])
        def safe():
            if 'username' in session:
                username = session['username']   
                return render_template('upload.html')
            return "You are not logged in <br><a href = '/login'></b>" + \
                "click here to log in</b></a>"
            
        
        #########################################################
        
        
        # Inserting/Adding contents in Safe Bank 
        
        @app.route('/upload_file', methods=['POST'])
        def upload_file():
            q = request.form['safename']
            r = request.files['file']
            r.save(os.path.join(app.config['UPLOAD_FOLDER1'],secure_filename(r.filename)))
            sqlformula="INSERT INTO safe(safeName,safeImage) VALUES (%s,%s)"
            room1=(q,(r.filename))
            db_cursor.execute(sqlformula,room1)
            mydb.commit()
            return redirect(url_for('safelist'))        
        
        ##########################################################
        
        
        # Deleting a particular entry from safe bank
        
        @app.route('/further',methods=['GET','POST'])
        def further():
            p=request.form['n']
            x=(p,)
            sql="""SELECT safeImage FROM safe WHERE safeID=%s """
            db_cursor.execute(sql,x)
            myresult = db_cursor.fetchall()
            sql_Delete_query = """Delete from safe where safeID = %s"""
            db_cursor.execute(sql_Delete_query,x)
            mydb.commit()
            root = os.path.join(os.path.expandvars(R'/home/ubuntu/opencv'),'static', 'safe',myresult[0][0] )
            os.remove(root)
            return redirect(url_for('safelist'))        
        
        #########################################################
        
        
        # Moving particular Identity from Safe Bank to Threat/UnSafe Bank
        
        @app.route('/move_file',methods=['GET','POST'])
        def move_file():
            p=request.form['no']
            dst_dir = '/home/ubuntu/opencv/static/threat'
            x=(p,)
            sql="""SELECT * FROM safe WHERE safeID=%s """
            db_cursor.execute(sql,x)
            myresult = db_cursor.fetchall()
            sqlformula="INSERT INTO unsafe(UnsafeName,UnsafeImage) VALUES (%s,%s)"
            room1=(myresult[0][1],myresult[0][2])
            db_cursor.execute(sqlformula,room1)
            sql_Delete_query = """Delete from safe where safeID = %s"""
            db_cursor.execute(sql_Delete_query,x)
            mydb.commit()
            print(myresult[0][2])
            root_src_dir = os.path.join('.','static/safe')
            root_target_dir = os.path.join('.','static/threat')

            operation= 'move' 
            file=myresult[0][2]
            for src_dir, dirs, files in os.walk(root_src_dir):
                dst_dir = src_dir.replace(root_src_dir, root_target_dir)
                if not os.path.exists(dst_dir):
                    os.mkdir(dst_dir)
                src_file = os.path.join(src_dir, file)
                if operation is 'copy':
                    shutil.copy(src_file, dst_dir)
                elif operation is 'move':
                    shutil.move(src_file,dst_dir)
            return redirect(url_for('safelist'))

        
        
        ##################################################################################################################
        
        
         
        # UnSAFE BANK FUNCTIONALIOTY
        #--------------------------------------------
        
        # Displaying UnSafe Bank Detail
        
        @app.route('/threatlist', methods=['GET', 'POST'])
        def threatlist():
            if 'username' in session:
                username = session['username']
               	db_cursor.execute('SELECT * FROM unsafe')
                table_rows = db_cursor.fetchall()
                df1 = pd.DataFrame(table_rows)
                list1=df1.values.tolist()
                print("lisss",table_rows)
                id=[]
                name=[]
                image=[]
                for i in range(len(table_rows)):
                    id.append(table_rows[i][0])
                    name.append(table_rows[i][1])
                    image.append(table_rows[i][2])
                lis={'id':id,'name':name,'image':image}
                return render_template('unsafedisp.html',tab=table_rows,ta=lis,len=len(table_rows))  
            
            return "You are not logged in <br><a href = '/login'></b>" + \
                "click here to log in</b></a>"
            
        
        ##############################################################
        
        
        # Inserting/Adding Form in UnSafe Bank 
        
        @app.route('/threat',methods=['GET','POST'])
        def threat():
            if 'username' in session:
                username = session['username']   
                return render_template('upload1.html')
            return "You are not logged in <br><a href = '/login'></b>" + \
            "click here to log in</b></a>"
            
        
        #############################################################
        
        
        # Inserting/Adding contents in UnSafe Bank 
        
        @app.route('/upload_file1', methods=['POST','GET'])
        def upload_file1():
            if request.method == "POST":
               q = request.form['unsafename']
               r = request.files['file']
               r.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(r.filename)))
               sqlformula="INSERT INTO unsafe(UnsafeName,UnsafeImage) VALUES (%s,%s)"
               room1=(q,(r.filename))
               db_cursor.execute(sqlformula,room1)
               mydb.commit()                
            return redirect(url_for('threatlist'))
        
        
        
        ##################################################################
       
        
        # Deleting a particular entry from Unsafe bank
       
        @app.route('/further1',methods=['GET','POST'])
        def further1():
            p=request.form['n']
            #print(p,type(p),p[0],p[1])
            x=(p,)
            sql="""SELECT UnsafeImage FROM unsafe WHERE UnsafeID=%s """
            db_cursor.execute(sql,x)
            myresult = db_cursor.fetchall()
            sql_Delete_query = """Delete from unsafe where UnsafeID = %s"""
            db_cursor.execute(sql_Delete_query,x)
            mydb.commit()
            root = os.path.join(os.path.expandvars(R'/home/ubuntu/opencv'),'static', 'threat',myresult[0][0] )
            os.remove(root)
            return redirect(url_for('threatlist'))        
            

        #################################################################
        
            
        # Moving particular Identity from UnSafe/Threat Bank to Safe Bank
        
        @app.route('/move',methods=['GET','POST'])
        def move():
            p=request.form['no']
            dst_dir = '/home/ubuntu/opencv/static'
            x=(p)
            db_cursor.execute("""SELECT * FROM unsafe WHERE UnsafeID=%s""",(x,))
            myresult = db_cursor.fetchall()
            sqlformula="INSERT INTO safe(safeName,safeImage) VALUES (%s,%s)"
            db_cursor.execute("""Delete from unsafe where UnsafeID = %s""",(x,))
            room1=(myresult[0][1],myresult[0][2])
            print("myresultaaaaaa",myresult)
            db_cursor.execute(sqlformula,room1)
            mydb.commit()
            root_src_dir = os.path.join('/home/ubuntu/opencv/static','threat')
            root_target_dir = os.path.join('/home/ubuntu/opencv/static','safe')

            operation= 'move'
            file=myresult[0][2]
            for src_dir, dirs, files in os.walk(root_src_dir):
                dst_dir = src_dir.replace(root_src_dir, root_target_dir)
                if not os.path.exists(dst_dir):
                    os.mkdir(dst_dir)
                src_file = os.path.join(src_dir, file)
                if operation is 'copy':
                    shutil.copy(src_file, dst_dir)
                elif operation is 'move':
                    shutil.move(src_file,dst_dir)

            return redirect(url_for('threatlist'))

            
        
        ################################################################################################################
        
        
         # UNKNOWN BANK FUNCTIONALITY
        #---------------------------------
        
        # Display Unknown Bank

        @app.route('/unknownlist', methods=['GET', 'POST'])
        def unknownlist():
            global list4
            
            #displaying unknown bank
            db_cursor.execute('SELECT * FROM unknown')
            table_rows = db_cursor.fetchall()
            df4 = pd.DataFrame(table_rows)
            list4=df4.values.tolist()
            if 'username' in session:
                username = session['username']
                db_cursor.execute('SELECT * FROM unknown')
                table_rows = db_cursor.fetchall()
                df4 = pd.DataFrame(table_rows)
                list4=df4.values.tolist()
                print("llllist4",list4)
                id=[]
                image=[] 
                for i in range(len(table_rows)):
                    id.append(table_rows[i][0])
                    image.append(table_rows[i][1])
                lis={'id':id,'image':image}
                return render_template('unknown.html',tab=table_rows,ta=lis,len=len(table_rows))  
            
            return "You are not logged in <br><a href = '/login'></b>" + \
                "click here to log in</b></a>"
        
        
        #######################################################################
        
        # Name for moving from unknown to safe
        
        @app.route('/unknownsafe',methods=['GET','POST'])
        def unknownsafe():
            global myresult1
            p=request.form['no']
            print(p)
            x=(p,)
            sql="""SELECT * FROM unknown WHERE id=%s """
            db_cursor.execute(sql,x)

            myresult1 = db_cursor.fetchall()
            x=myresult1[0][1]
            return redirect(url_for('moveunknown'))
        
        
        #######################################################################
        
        # Asking for Name for Safe bank
        
        @app.route('/moveunknown',methods=['GET','POST'])
        def moveunknown():
            return render_template('unknownmove.html')
        
        
        #######################################################################
        
        # Move from unknown to safe bank
        
        @app.route('/movetosafe',methods=['GET','POST'])
        def movetosafe():
            p=request.form['name']
            dst_dir = '/home/ubuntu/opencv/static/safe'
            i=(myresult1[0][0],)
            l=(2,)
            sqlformula="INSERT INTO safe(safeName,safeImage) VALUES (%s,%s)"
            sql_Delete_query = """Delete from unknown where id = %s"""
            db_cursor.execute(sql_Delete_query,i)
            room1=(p,myresult1[0][1])
            db_cursor.execute(sqlformula,room1)
            mydb.commit()
            root_src_dir = os.path.join('.','/static/unknown')
            root_target_dir = os.path.join('.','static/safe')
            operation= 'move'
            file=myresult1[0][1]
            for src_dir, dirs, files in os.walk(root_src_dir):
                dst_dir = src_dir.replace(root_src_dir, root_target_dir)
                if not os.path.exists(dst_dir):
                    os.mkdir(dst_dir)               
                src_file = os.path.join(src_dir, file)
                if operation is 'copy':
                    shutil.copy(root_src_dir, dst_dir)
                elif operation is 'move':
                    shutil.move(src_file,dst_dir)
            return redirect(url_for('unknownlist'))
            
        ##################################################################################
        # Inserting/Adding Form in UnSafe Bank 
        
        @app.route('/unknown',methods=['GET','POST'])
        def unknown():
            if 'username' in session:
                username = session['username']   
                return render_template('upload2.html')
            return "You are not logged in <br><a href = '/login'></b>" + \
                "click here to log in</b></a>"
            
        ##################################################################################
        @app.route('/upload_file2', methods=['POST','GET'])
        def upload_file2():
            if request.method == "POST":
                p = request.form['unknownid']
                r = request.files['file1']
                r.save(os.path.join(app.config['UPLOAD_FOLDER2'],secure_filename(r.filename)))
                sqlformula="INSERT INTO unknown(id,image) VALUES (%s,%s)"
                room1=(p,(r.filename))
                db_cursor.execute(sqlformula,room1)
                mydb.commit()             
            return redirect(url_for('index'))
        ##########################################################################
        
        # Name for moving from unknown to unsafe
        
        @app.route('/unknownunsafe',methods=['GET','POST'])
        def unknownunsafe():
            global myresult2
            p=request.form['n']
            x=(p,)
            sql="""SELECT * FROM unknown WHERE id=%s """
            db_cursor.execute(sql,x)
            myresult2 = db_cursor.fetchall()
            x=myresult2[0][1]
            return redirect(url_for('move1unknown'))
        #######################################################################
        
        # Asking for name for Unsafe Bank
        
        @app.route('/move1unknown',methods=['GET','POST'])
        def move1unknown():
            return render_template('unknownmove1.html')
        
        #######################################################################
        
        # Move from unknown to unsafe
        
        @app.route('/movetounsafe',methods=['GET','POST'])
        def movetounsafe():
            p=request.form['name']
            print(p,myresult2[0][0])
            dst_dir = '/home/ubuntu/opencv/static/threat'
            i=(myresult2[0][0],)
            l=(2,)
            sqlformula="INSERT INTO unsafe(unsafeName,unsafeImage) VALUES (%s,%s)"
            sql_Delete_query = """Delete from unknown where id = %s"""
            db_cursor.execute(sql_Delete_query,i)
            room1=(p,myresult2[0][1])
            db_cursor.execute(sqlformula,room1)
            mydb.commit()
            
            #root = os.path.join(os.path.expanduser('~'),'opencv', 'safe',myresult[0][0] )
            #os.remove(root)
            root_src_dir = os.path.join('.','static/unknown')
            root_target_dir = os.path.join('.','static/threat')

            operation= 'move' # 'copy' or 'move'
            file=myresult2[0][1]
            for src_dir, dirs, files in os.walk(root_src_dir):
                dst_dir = src_dir.replace(root_src_dir, root_target_dir)
                if not os.path.exists(dst_dir):
                    os.mkdir(dst_dir)
                #for file_ in files:
                src_file = os.path.join(src_dir, file)
                #dst_file = os.path.join(dst_dir, file_)
                #if os.path.exists(dst_file):
                    #os.remove(dst_file)
                if operation is 'copy':
                    shutil.copy(root_src_dir, dst_dir)
                elif operation is 'move':
                    shutil.move(src_file,dst_dir)
            return redirect(url_for('unknownlist'))                
        ####################################################################################################################        
        # Deleting Data From Unknown Table   
        @app.route('/delete',methods=['GET','POST'])
        def delete():
            p=request.form['n1']
            x=(p,)
            sql="""SELECT image FROM unknown WHERE id=%s """
            db_cursor.execute(sql,x)
            myresult = db_cursor.fetchall()
            sql_Delete_query = """Delete from unknown where id = %s"""
            db_cursor.execute(sql_Delete_query,x)
            mydb.commit()
            root = os.path.join(os.path.expandvars(R'/home/ubuntu/opencv'),'static', 'unknown',myresult[0][0] )
            os.remove(root)
            return redirect(url_for('unknownlist'))
#################################################################################################################### 
        # Displaying Camera Details
        @app.route('/display', methods=['GET', 'POST'])
        def display():
            if 'username' in session:
                username = session['username']
                db_cursor.execute('SELECT * FROM camdetail')
                table_rows = db_cursor.fetchall()
                df1 = pd.DataFrame(table_rows)
                p=df1.values.tolist()
                return render_template('index1.html',tab=p)
            return "You are not logged in <br><a href = '/login'></b>" + \
                "click here to log in</b></a>"
        ####################################################################
        
        
        # INsert/Add form 
            
        @app.route('/insert',methods=['GET', 'POST'])
        def insert():
            """Video streaming home page."""
            return render_template('insert.html')

        
        #####################################################################
        
        
        
        # Insering/Adding Data into camdetail Table
        
        @app.route('/handle_data', methods=['GET','POST'])
        def handle_data():
            q = request.form['Location']
            r = request.form['ip_address']
            s=request.form['camera_info']
            t=request.form['status']
            sqlformula="INSERT INTO camdetail(Location,ip_address,camera_info,status) VALUES (%s,%s,%s,%s)"
            room1=(q,r,s,t)
            db_cursor.execute(sqlformula,room1)
            mydb.commit()
            return redirect(url_for('display'))
        ##################################################################
        
        
        # Update form
        
        @app.route('/update', methods=['POST'])
        def update():
            if 'username' in session:
                username = session['username']   
                s=request.form['n']
                x=(s,)
                sql="""SELECT * FROM camdetail WHERE sno=%s """
                db_cursor.execute(sql,x)
                data = db_cursor.fetchall()  
                data1 = pd.DataFrame(data)
                data2=data1.values.tolist() 
                return render_template('insert1.html',t=data2)
            return "You are not logged in <br><a href = '/login'></b>" + \
                "click here to log in</b></a>"
            
        
        
        ###################################################################
        
        
        # Updating data of camdetail Table
        
        @app.route('/update1', methods=['GET','POST'])
        def update1():
            if request.method == "POST":
                sn = request.form['no']
                q = request.form['Location']
                r = request.form['ip_address']
                s= request.form['camera_info']
                t= request.form['status']           
                sql = "UPDATE camdetail SET Location =%s , ip_address=%s , camera_info=%s, status=%s WHERE sno = %s"
                val=(q,r,s,t,sn)
                db_cursor.execute(sql,val)
                mydb.commit()
                return redirect(url_for('chart'))        
        ###################################################################################################################
        
        
        
        # Displaying Advanced setting page
        
        @app.route('/settingdisp',methods=['GET','POST'])
        def settingdisp():
            """Video streaming home page."""
            if 'username' in session:
                username = session['username']
                return render_template('setdisp.html',tab=list3)
            return "You are not logged in <br><a href = '/login'></b>" + \
                "click here to log in</b></a>"
        
        
        ######################################################################
        
        
        # Updation Form

        @app.route('/settings',methods=['GET', 'POST'])

        def settings():
            if 'username' in session:
                username = session['username']   
                return render_template('settings1.html',tab=p)
            return "You are not logged in <br><a href = '/login'></b>" + \
                "click here to log in</b></a>"
               
            
        
        
        #####################################################################
            
        
        # Updating data of Advancde Setting Table
        
        @app.route('/handle_settings', methods=['POST'])
        def handle_settings():
            global rows,column,c2,d2,e2
            a1 = request.form['drop1']
            b1 = request.form['drop2']
            c1 = request.form['drop3']
            d1 = request.form['drop4']
             
            rows = request.form['rows']
            column = request.form['columns']

            b2 = request.form['notification']
            c2 = request.form['age']
            d2 = request.form['gender']
            e2 = request.form['color']
            
            e=request.form['email']
            x=b2[0]
            x1=b2[0:2]
            x2=b2[0:3]
            x3=b2[1:3]
            id='1'            

            if x is '1':
                #sql = "INSERT INTO settings(Safe_color,Unsafe_color,Weapon_color,Smoke_Fire_color,Notification_Smoke_Fire,Notification_Unsafe,Notification_Weapon,Notifiction_None,Ro,col,email,age,gender,Colours_of_clothes)VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                #val=(a1,b1,c1,d1,x,'-','-','-',rows,column,e,c2,d2,e2)
                sql = "UPDATE settings SET Safe_color=%s, Unsafe_color=%s,Weapon_color=%s,Smoke_Fire_color=%s,Notification_Smoke_Fire=%s,Notification_Unsafe=%s,Notification_Weapon=%s,Notification_None=%s,Ro=%s,col=%s,email=%s,age=%s,gender=%s,Colours_of_clothes=%s WHERE Id=%s"
                val=(a1,b1,c1,d1,x,'-','-','-',rows,column,e,c2,d2,e2,id)
                db_cursor.execute(sql,val)
                mydb.commit()
        
            elif x is '2':

                sql = "UPDATE settings SET Safe_color=%s, Unsafe_color=%s,Weapon_color=%s,Smoke_Fire_color=%s,Notification_Smoke_Fire=%s,Notification_Unsafe=%s,Notification_Weapon=%s,Notification_None=%s,Ro=%s,col=%s,email=%s,age=%s,gender=%s,Colours_of_clothes=%s WHERE Id=%s"
                val=(a1,b1,c1,d1,'-',x,'-','-',rows,column,e,c2,d2,e2,id)
                db_cursor.execute(sql,val)
                mydb.commit()

            elif x is '3':

                sql = "UPDATE settings SET Safe_color=%s, Unsafe_color=%s,Weapon_color=%s,Smoke_Fire_color=%s,Notification_Smoke_Fire=%s,Notification_Unsafe=%s,Notification_Weapon=%s,Notifiction_None=%s,Ro=%s,col=%s,email=%s,age=%s,gender=%s,Colours_of_clothes=%s WHERE Id=%s"
                val=(a1,b1,c1,d1,'-','-',x,'-',rows,column,e,c2,d2,e2,id)
                db_cursor.execute(sql,val)
                mydb.commit()
            else:
                sql = "UPDATE settings SET Safe_color=%s, Unsafe_color=%s,Weapon_color=%s,Smoke_Fire_color=%s,Notification_Smoke_Fire=%s,Notification_Unsafe=%s,Notification_Weapon=%s,Notifiction_None=%s,Ro=%s,col=%s,email=%s,age=%s,gender=%s,Colours_of_clothes=%s WHERE Id=%s"
                val=(a1,b1,c1,d1,'-','-','-',x,rows,column,e,c2,d2,e2,id)
                db_cursor.execute(sql,val)
                mydb.commit()
            
            if x2 == ['1','2','3']:
                if x2[0] is '1' and x2[1] is '2' and x2[2] is '3':
                    sql = "UPDATE settings SET Safe_color=%s, Unsafe_color=%s,Weapon_color=%s,Smoke_Fire_color=%s,Notification_Smoke_Fire=%s,Notification_Unsafe=%s,Notification_Weapon=%s,Notifiction_None=%s,Ro=%s,col=%s,email=%s,age=%s,gender=%s,Colours_of_clothes=%s WHERE Id=%s"
                    val=(a1,b1,c1,d1,x2[0],x2[1],x2[2],'-',rows,column,e,c2,d2,e2,id)
                    db_cursor.execute(sql,val)
                    mydb.commit()

            if x1 == ['1','2']:
                if x1[0] is '1' and x1[1] is '2':
                    sql = "UPDATE settings SET Safe_color=%s, Unsafe_color=%s,Weapon_color=%s,Smoke_Fire_color=%s,Notification_Smoke_Fire=%s,Notification_Unsafe=%s,Notification_Weapon=%s,Notifiction_None=%s,Ro=%s,col=%s,email=%s,age=%s,gender=%s,Colours_of_clothes=%s WHERE Id=%s"
                    val=(a1,b1,c1,d1,x[0],x1[1],'-','-',rows,column,e,c2,d2,e2,id)
                    db_cursor.execute(sql,val)
                    mydb.commit()
        
            if x3 == ['2','3']:
                if x3[0] is '2' and x3[1] is '3':
                    sql = "UPDATE settings SET Safe_color=%s, Unsafe_color=%s,Weapon_color=%s,Smoke_Fire_color=%s,Notification_Smoke_Fire=%s,Notification_Unsafe=%s,Notification_Weapon=%s,Notifiction_None=%s,Ro=%s,col=%s,email=%s,age=%s,gender=%s,Colours_of_clothes=%s WHERE Id=%s"
                    val=(a1,b1,c1,d1,'-',x3[0],x3[1],'-',rows,column,e,c2,d2,e2,id)
                    db_cursor.execute(sql,val)
                    mydb.commit()
            return redirect(url_for('index'))

####################################################################################################################
        
        
        @app.route('/video_feed/<int:url1>')
        def video_feed(url1):
                        return Response((detectmotion(url1)),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

else:
        
        
        # Login session
        
        @app.route('/',methods=['GET','POST'])
        def index():
            if 'username' in session:
                username = session['username']
                return redirect(url_for('settingdisp'))
            return "You are not logged in <br><a href = '/login'></b>" + \
                "click here to log in</b></a>"
        
        ####################################################################################################################
        
        #login page
        
        @app.route('/login', methods = ['GET', 'POST'])
        def login():
            if request.method == "POST":
                username = request.form['username']
                password = request.form['password'].encode('utf-8')
                cur = mydb.cursor(dictionary=True)
                cur.execute("SELECT * FROM users WHERE username=%s",(username,))
                user = cur.fetchone()
                cur.close()
                
                if len(user) > 0:
                    if bcrypt.hashpw(password,user["password"].encode('utf-8')) == user["password"].encode('utf-8'):
                    #Create a user login session
                        session['name'] = user['name']
                        session['username'] = user['username']
                        return redirect(url_for("index"))
                    else:
                        return "Enter Password  Or Username Not Match!"
                else:
                    return "Enter Password  Or Username Not Match!"
            else:
                return render_template('login.html')

        ##################################################################################################################
        #Register Page

        @app.route('/register',methods=["POST","GET"])
        def register():
            if request.method=="GET":
                redirect('login')
            else:
                name = request.form['name']
                email = request.form['email']
                username = request.form['username']
                password = request.form['password'].encode('utf-8')
                hash_password = bcrypt.hashpw(password,bcrypt.gensalt())
                cur=mydb.cursor()
                cur.execute("INSERT INTO users(name,username,email,password) VALUES(%s,%s,%s,%s)",
                    (name,username,email,hash_password,))
                mydb.commit()
                return redirect(url_for("login"))
            return render_template("register.html")


        
        
        
        # Displaying Camera Details

        @app.route('/display', methods=['GET', 'POST'])
        def display():
            return render_template('index1.html',tab=p)
        
        
        ############################################################


        # INsert/Add form 
        
        @app.route('/insert',methods=['GET', 'POST'])

        def insert():
            """Video streaming home page."""
            return render_template('insert.html')
        
        
        ############################################################
        
        
        # Insering/Adding Data into camdetail Table

        @app.route('/handle_data', methods=['GET','POST'])
        def handle_data():
            q = request.form['Location']
            r = request.form['ip_address']
            s=request.form['camera_info']
            t=request.form['status']
            #u=request.form['']
            sqlformula="INSERT INTO camdetail(Location,ip_address,camera_info,status) VALUES (%s,%s,%s,%s)"
            room1=(q,r,s,t)
            db_cursor.execute(sqlformula,room1)
            mydb.commit()
            return redirect(url_for('index'))
        
        
        #################################################################################################################
        
        
        # Displaying Advanced setting page
        
        @app.route('/settingdisp',methods=['GET','POST'])
        def settingdisp():
            """Video streaming home page."""
            if 'username' in session:
                username = session['username']
                
                return render_template('setdisp.html',tab=list3)
            return "You are not logged in <br><a href = '/login'></b>" + \
                "click here to log in</b></a>"
        
        
        ######################################################################
        
        
        # Updation Form

        @app.route('/settings',methods=['GET', 'POST'])

        def settings():              
            return render_template('settings1.html',tab=p)
        
        
        #####################################################################
        
        
        
        # Updating data of Advancde Setting Table

        @app.route('/handle_settings', methods=['POST'])
        def handle_settings():
            global rows,column
            a1 = request.form['drop1']
            b1 = request.form['drop2']
            c1 = request.form['drop3']
            d1 = request.form['drop4']
             
            rows = request.form['rows']
            column = request.form['columns']

            b2 = request.form['fire']
            c2 = request.form['age']
            d2 = request.form['gender']
            e2 = request.form['color']
            
            e=request.form['email']
            x=b2
            id='1'
        
            if x is '1':
                #sql = "INSERT INTO settings(Safe_color,Unsafe_color,Weapon_color,Smoke_Fire_color,Notification_Smoke_Fire,Notification_Unsafe,Notification_Weapon,Notifiction_None,Ro,col,email,age,gender,Colours_of_clothes)VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                #val=(a1,b1,c1,d1,x,'-','-','-',rows,column,e,c2,d2,e2)

                sql = "UPDATE settings SET Safe_color=%s, Unsafe_color=%s,Weapon_color=%s,Smoke_Fire_color=%s,Notification_Smoke_Fire=%s,Notification_Unsafe=%s,Notification_Weapon=%s,Notifiction_None=%s,Ro=%s,col=%s,email=%s,age=%s,gender=%s,Colours_of_clothes=%s WHERE Id=%s"
                val=(a1,b1,c1,d1,x,'-','-','-',rows,column,e,c2,d2,e2,id)
                db_cursor.execute(sql,val)
                mydb.commit()
            
            elif x is '2':

                sql = "UPDATE settings SET Safe_color=%s, Unsafe_color=%s,Weapon_color=%s,Smoke_Fire_color=%s,Notification_Smoke_Fire=%s,Notification_Unsafe=%s,Notification_Weapon=%s,Notifiction_None=%s,Ro=%s,col=%s,email=%s,age=%s,gender=%s,Colours_of_clothes=%s WHERE Id=%s"
                val=(a1,b1,c1,d1,'-',x,'-','-',rows,column,e,c2,d2,e2,id)
                db_cursor.execute(sql,val)
                mydb.commit()

            elif x is '3':

                sql = "UPDATE settings SET Safe_color=%s, Unsafe_color=%s,Weapon_color=%s,Smoke_Fire_color=%s,Notification_Smoke_Fire=%s,Notification_Unsafe=%s,Notification_Weapon=%s,Notifiction_None=%s,Ro=%s,col=%s,email=%s,age=%s,gender=%s,Colours_of_clothes=%s WHERE Id=%s"
                val=(a1,b1,c1,d1,'-','-',x,'-',rows,column,e,c2,d2,e2,id)
                db_cursor.execute(sql,val)
                mydb.commit()

            else:

                sql = "UPDATE settings SET Safe_color=%s, Unsafe_color=%s,Weapon_color=%s,Smoke_Fire_color=%s,Notification_Smoke_Fire=%s,Notification_Unsafe=%s,Notification_Weapon=%s,Notifiction_None=%s,Ro=%s,col=%s,email=%s,age=%s,gender=%s,Colours_of_clothes=%s WHERE Id=%s"
                val=(a1,b1,c1,d1,'-','-','-',x,rows,column,e,c2,d2,e2,id)
                db_cursor.execute(sql,val)
                mydb.commit()
            
            if x2 == ['1','2','3']:
                if x2[0] is '1' and x2[1] is '2' and x2[2] is '3':
                    sql = "UPDATE settings SET Safe_color=%s, Unsafe_color=%s,Weapon_color=%s,Smoke_Fire_color=%s,Notification_Smoke_Fire=%s,Notification_Unsafe=%s,Notification_Weapon=%s,Notifiction_None=%s,Ro=%s,col=%s,email=%s,age=%s,gender=%s,Colours_of_clothes=%s WHERE Id=%s"
                    val=(a1,b1,c1,d1,x2[0],x2[1],x2[2],'-',rows,column,e,c2,d2,e2,id)
                    db_cursor.execute(sql,val)
                    mydb.commit()

            if x1 == ['1','2']:
                if x1[0] is '1' and x1[1] is '2':
                    sql = "UPDATE settings SET Safe_color=%s, Unsafe_color=%s,Weapon_color=%s,Smoke_Fire_color=%s,Notification_Smoke_Fire=%s,Notification_Unsafe=%s,Notification_Weapon=%s,Notifiction_None=%s,Ro=%s,col=%s,email=%s,age=%s,gender=%s,Colours_of_clothes=%s WHERE Id=%s"
                    val=(a1,b1,c1,d1,x[0],x1[1],'-','-',rows,column,e,c2,d2,e2,id)
                    db_cursor.execute(sql,val)
                    mydb.commit()
        
            if x3 == ['2','3']:
                if x3[0] is '2' and x3[1] is '3':
                    sql = "UPDATE settings SET Safe_color=%s, Unsafe_color=%s,Weapon_color=%s,Smoke_Fire_color=%s,Notification_Smoke_Fire=%s,Notification_Unsafe=%s,Notification_Weapon=%s,Notifiction_None=%s,Ro=%s,col=%s,email=%s,age=%s,gender=%s,Colours_of_clothes=%s WHERE Id=%s"
                    val=(a1,b1,c1,d1,'-',x3[0],x3[1],'-',rows,column,e,c2,d2,e2,id)
                    db_cursor.execute(sql,val)
                    mydb.commit()
            return redirect(url_for('index'))
     
if __name__ == '__main__':
    app.run(host='0.0.0.0',threaded = True)
