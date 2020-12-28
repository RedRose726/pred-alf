from flask import Flask,render_template,request
from DBConnection import Db
from flask.globals import session

app = Flask(__name__)
app.secret_key="abcd"

@app.route('/')
def log():
    return render_template('form.html')

@app.route('/login',methods=['post'])
def login():
    try:
        c=Db()
        username=request.form['username']
        pswrd=request.form['pass']
        qry="select * from login where username='"+username+"' and password='"+pswrd+"'"
        result=c.selectOne(qry)

        if result!=None:
            type = result['type']
            session["userid"] = result["lid"]
            if type=='admin':
                    return render_template('admin/adminHome.html')
            elif type=='user':
                return render_template('user/user home.html')
            elif type == 'blocked':
                return '<script>alert("Login restricted!! You have been blocked by the Admin");window.location="/";</script>'

        if result is None:
            return '<script>alert("incorrect username or pssword");window.location="/";</script>'
    except Exception as e:
        print(str(e))

@app.route('/logout')
def logout():
    session.pop("userid")
    session["userid"]=None
    return render_template('form.html')

@app.route('/reg')
def reg():
    return render_template('registration.html')

@app.route('/regist',methods=['post'])
def regist():
    c=Db()
    fname = request.form['textfield']
    gender=request.form['radio']
    date=request.form['textfield3']
    mail=request.form['textfield4']
    pswrd=request.form['textfield2']
    confpswrd=request.form['textfield5']
    if pswrd==confpswrd:
        qr="insert into login (username,password,type)values('"+mail+"','"+confpswrd+"','user')"
        res1=c.insert(qr)
        qry="insert into user (loginid,name,gender,dob,mail)values('"+str(res1)+"','"+fname+"','"+gender+"','"+date+"','"+mail+"')"
        res=c.insert(qry)
        return '''<script>alert('User registered successfully!!!');window.location='/reg';</script>'''

    else:
        return '''<script>alert('passwords not matching');window.location='/';</script>'''



@app.route('/chpswrd')
def chpswrd():
    c=Db()
    x = session['userid']
    if session["userid"] != None:
        y = "select * from login where lid='" + str(x) + "'"
        r = c.selectOne(y)
        if r is not None:
            type = r['type']
            if type == 'admin':
                return render_template('admin/change pswrd.html')
            else:
                return render_template('user/user pswrd.html')
    else:
        return render_template('form.html')



@app.route('/change',methods=['post'])
def change():
    c=Db()
    crntpswrd = request.form['textfield']
    newpswrd=request.form['textfield2']
    cnfrmpswrd=request.form['textfield3']
    x=session['userid']

    if newpswrd == cnfrmpswrd:
        qry = "select password from login where lid='"+str(x)+"'"
        res = c.selectOne(qry)
        if res is not None:
            if res['password']==crntpswrd:
                q="update login set password='"+newpswrd+"' where lid='"+str(x)+"'"
                c.update(q)
                y="select * from login where lid='"+str(x)+"'"
                r=c.selectOne(y)
                if r is not None:
                    type=r['type']
                    if type=='admin':
                        return adm_home()
                    else:
                        return user_home()
            else:
                return '''<script>alert('incorrect current password');window.location='/chpswrd';</script>'''
    else:
        return '''<script>alert('passwords not matching');window.location='/chpswrd';</script>'''





# ....................................admin...............................................................


@app.route('/adm_home')
def adm_home():
    if session["userid"] != None:
        return render_template('admin/adminHome.html')
    else:
        return render_template('form.html')


@app.route('/adm_users')
def adm_users():
    c=Db()
    if session["userid"] != None:
        qry = "select user.*,login.* from login,user where user.loginid=login.lid"
        res = c.select(qry)
        return render_template('admin/allUsers.html', data=res)
    else:
        return render_template('form.html')


@app.route('/rem/<id>')
def rem(id):
    c=Db()
    res=c.delete("delete from login where lid='"+id+"'")
    c.delete("delete from user where loginid='"+id+"'")
    return adm_users()

@app.route('/block/<id>/<type>')
def block(id,type):
    c=Db()
    q="update login set type='"+type+"' where lid='"+id+"'"
    res=c.update(q)
    return adm_users()


@app.route('/adm_cmplnt')
def adm_cmplnt():
    a = Db()
    if session["userid"] != None:
        q = "select complaint.*,user.* from user,complaint where complaint.clid=user.loginid"
        r = a.select(q)
        return render_template('admin/adminComplaintform.html', cmplnts=r)
    else:
        return render_template('form.html')


@app.route('/adm_viewCmplnt/<id>/<cm>')
def adm_viewCmplnt(id,cm):
    rr=cm
    if session["userid"] != None:
        session["id"] = id
        return render_template('admin/viewComplaint.html', rr=cm)
    else:
        return render_template('form.html')


@app.route('/reply',methods=['post'])
def reply():
    c=Db()
    response = request.form['textfield']
    q="update complaint set response='"+response+"' where cmpid='"+session["id"]+"'"
    r=c.update(q)
    return adm_cmplnt()


@app.route('/adm_feed')
def adm_feed():
    c = Db()
    if session["userid"] != None:
        qry = "select feedback.*,user.* from user,feedback where user.loginid=feedback.flid"
        rslt = c.select(qry)
        return render_template('admin/adminFeedback.html', feeds=rslt)
    else:
        return render_template('form.html')



#....................................user......................................
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.externals import joblib


def accuracyPredictor(X_train,X_test,Y_train,Y_test):
    rfc = RandomForestClassifier(n_estimators=250, criterion='entropy', random_state=0)
    rfc.fit(X_train, Y_train)
    rfc_predict = rfc.predict(X_test)
    accuracy = accuracy_score(Y_test, rfc_predict)
    print("The accuracy of RF algorithm is: ", str(accuracy * 100), "%")
    cm = confusion_matrix(Y_test, rfc_predict)
    print("CM : ", cm)
    rocauc = roc_auc_score(Y_test, rfc_predict)
    print("The roc of RF algorithm is: ", rocauc)

    lr = LogisticRegression(max_iter=500, random_state=0)
    lr.fit(X_train, Y_train)
    lr_predict = lr.predict(X_test)
    accuracy3 = accuracy_score(Y_test, lr_predict)
    print("The accuracy of LR algorithm is: ", str(accuracy3 * 100), "%")
    cm = confusion_matrix(Y_test, lr_predict)
    print("CM : ", cm)
    rocauc = roc_auc_score(Y_test, lr_predict)
    print("The roc of LR algorithm is: ", rocauc)

    dt = DecisionTreeClassifier(criterion="gini", random_state=0, min_samples_leaf=5)
    dt.fit(X_train, Y_train)
    dt_predict = dt.predict(X_test)
    accuracy2 = accuracy_score(Y_test, dt_predict)
    print("The accuracy of DT algorithm is: ", str(accuracy2 * 100), "%")
    cm = confusion_matrix(Y_test, dt_predict)
    print("CM : ", cm)
    rocauc = roc_auc_score(Y_test, dt_predict)
    print("The roc of DT algorithm is: ", rocauc)

    return str(accuracy3*100)


@app.route('/user_home')
def user_home():
    if session["userid"]!= None:
        return render_template('user/user home.html')
    else:
        return render_template('form.html')

@app.route('/user_profile')
def user_profile():
    c=Db()
    x=session['userid']
    if session["userid"] != None:
        q = "select * from user where loginid='" + str(x) + "'"
        result = c.selectOne(q)
        return render_template('user/my profile.html', data=result)
    else:
        return render_template('form.html')


@app.route('/user_det')
def user_det():
    if session["userid"] != None:
        return render_template('user/upload details.html')
    else:
        return render_template('form.html')


@app.route('/det',methods=['post'])
def det():
    c=Db()
    x=session["userid"]
    age = int(request.form['textfield'])
    gender = int(request.form['radio'])
    totBilirubin = float(request.form['textfield3'])
    dirBilirubin = float(request.form['textfield4'])
    alkPhosphotase = int(request.form['textfield5'])
    alamineAminotransferase = int(request.form['textfield6'])
    aspartateAminotransferase = int(request.form['textfield7'])
    totalProteins = float(request.form['textfield8'])
    albumin = float(request.form['textfield9'])
    albglobRatio = float(request.form['textfield10'])

    testdata = [[age, gender, totBilirubin, dirBilirubin, alkPhosphotase, alamineAminotransferase, aspartateAminotransferase,
         totalProteins, albumin, albglobRatio]]

    balance_data = pd.read_csv('C:/Users/LENOVA/PycharmProjects/mainpro/static/indian_liver_patient(origin).csv', sep=',')
    balance_data['Dataset'] = balance_data['Dataset'].map({2: 0, 1: 1})
    balance_data['Albumin_and_Globulin_Ratio'].fillna(value=0, inplace=True)
    X = balance_data.iloc[:, :-1].values  # seperating result from other datas
    Y = balance_data.iloc[:, -1].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)  # training n testing data


    to_predict = np.array(testdata)
    print("Input: ",to_predict)
    loaded_model1 = joblib.load('C:/Users/LENOVA/PycharmProjects/mainpro/lr.pkl')  # load the model from the file
    result = loaded_model1.predict(to_predict)  # use the loaded model to make predictions
    print("Result in LR: ", result)
    loaded_model2 = joblib.load('C:/Users/LENOVA/PycharmProjects/mainpro/rfc2.pkl')  # load the model from the file
    result2 = loaded_model2.predict(to_predict)  # use the loaded model to make predictions
    print("Result in RFC: ", result2)
    loaded_model3 = joblib.load('C:/Users/LENOVA/PycharmProjects/mainpro/dt2.pkl')  # load the model from the file
    result3 = loaded_model3.predict(to_predict)  # use the loaded model to make predictions
    print("Result in DT: ", result3)

    score=accuracyPredictor(X_train,X_test,Y_train,Y_test)
    if int(result)==1:
        result = 'ALF'
        prediction = 'The patient may have Advanced Liver Fibrosis. The probability of it is 71.23287671232876%'
    else:
        result = 'No ALF'
        prediction='The patient may not have Advanced Liver Fibrosis. The probability of it is 71.23287671232876%'

    q = "insert into details (ulid,result,udate,inputs) values('" + str(x) + "','"+result+"',curdate(),'"+str(testdata)+"')"
    r = c.insert(q)
    return render_template('user/upload details.html',predict=prediction)


@app.route('/result_history')
def result_history():
    c=Db()
    x=session["userid"]
    if session["userid"] != None:
        qry = "select * from details where ulid='"+str(x)+"'"
        res = c.select(qry)
        return render_template('user/resultHistory.html', details=res)
    else:
        return render_template('form.html')


@app.route('/delete/<id>')
def delete(id):
    c=Db()
    c.delete("delete from details where upid='"+id+"'")
    return result_history()


@app.route('/user_cmplnt')
def user_cmplnt():
    c = Db()
    x=session["userid"]
    if session["userid"] != None:
        qry = "select * from complaint where clid='" + str(x) + "'"
        rslt = c.select(qry)
        return render_template('user/user complaint form.html', ucmplnts=rslt)
    else:
        return render_template('form.html')


@app.route('/user_newcmplnt')
def user_newcmplnt():
    if session["userid"] != None:
        return render_template('user/new complaint.html')
    else:
        return render_template('form.html')


@app.route('/cmplnt',methods=['post'])
def cmplnt():
    c=Db()
    x=session["userid"]
    complaint = request.form['textfield2']
    q="insert into complaint (clid,complaint,response,cdate) values('"+str(x)+"','"+complaint+"','pending',curdate())"
    r=c.insert(q)
    return '''<script>alert('Your complaint placed successfully');window.location='/user_newcmplnt';</script>'''


@app.route('/user_feed')
def user_feed():
    if session["userid"] != None:
        return render_template('user/user feedback.html')
    else:
        return render_template('form.html')


@app.route('/fd',methods=['post'])
def fd():
    c = Db()
    x = session["userid"]
    feedback = request.form['textfield']
    q = "insert into feedback (flid,feedback,fdate) values('" + str(x) + "','" + feedback + "',curdate())"
    r = c.insert(q)
    return '''<script>alert('Thank you for the feedback!!!');window.location='/user_feed';</script>'''


if __name__ == '__main__':
    app.run(debug=True)
