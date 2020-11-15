from flask import Flask,render_template,request
app = Flask(__name__)
import pickle

file = open('ML_model.pkl','rb')
clf = pickle.load(file)
file.close()
@app.route('/', methods=["GET","POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        Fever = int(myDict['Fever'])
        Age = int(myDict['Age'])
        Pain = int(myDict['Pain'])
        Runny_Nose = int(myDict['Runny_Nose'])
        Difficulty_Breath = int(myDict['Difficulty_Breath'])

       
        inputfeatures = [Fever,Pain,Age,Runny_Nose,Difficulty_Breath]
        infprob = clf.predict_proba([inputfeatures])[0][1]
        print(infprob)
        return render_template('show.html',inf=round(infprob*100))
    return render_template('index.html')
    

if __name__ == "__main__":
    app.run(debug=True)