from django.shortcuts import render
import pandas as pd
from .models import emlploy_data
from .models import employ_data_hist
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db.models.functions import Cast
from django.db.models import IntegerField
import numpy as np
import joblib
from django.core.paginator import Paginator

#importing joblib files we saved 
preprocessor=joblib.load(r"E:\data\employe_model\ml_model\catagorical_columns_encode.joblib")
pipeline_d=joblib.load(r"E:\data\employe_model\ml_model\employ_attrition_pipeline.joblib")
pt_distance=joblib.load(r"E:\data\employe_model\ml_model\DistanceFromHome.joblib")
pt_income=joblib.load(r"E:\data\employe_model\ml_model\MonthlyIncome.joblib")
pt_companies=joblib.load(r"E:\data\employe_model\ml_model\NumCompaniesWorked.joblib")
pt_total_years=joblib.load(r"E:\data\employe_model\ml_model\total_working_years_power.joblib")
pt_company_years=joblib.load(r"E:\data\employe_model\ml_model\years_at_company_power.joblib")

def dashboard (request):
    #total headcount
    total_headcount=emlploy_data.objects.all()
    total_headcount=len(total_headcount)

    #active employyes
    active_employ=emlploy_data.objects.all().exclude(Attrition="Yes").values_list('Attrition',flat=True)
    active_employ=len(active_employ)

    #terminated employyes
    terminated_employ=emlploy_data.objects.all().exclude(Attrition="No").values_list('Attrition',flat=True)
    terminated_employ=len(terminated_employ)

    #attrition rate
    attrition_rate=(terminated_employ/active_employ)*100
    attrition_rate=round(attrition_rate,2)

    #attrition by department chart
    dept_labels=emlploy_data.objects.all().values_list("JobRole",flat=True).distinct()
    print(dept_labels)
    dept_values=[]
    for i in dept_labels:
        dept_values.append(emlploy_data.objects.filter(JobRole=i,Attrition='Yes').values_list().count())
    print(dept_values)

    #gender pie chart
    active_males=emlploy_data.objects.filter(Gender='Male',Attrition='No').values_list().count()
    active_females=emlploy_data.objects.filter(Gender='Female',Attrition='No').values_list().count()
    terminated_male=emlploy_data.objects.filter(Gender='Male',Attrition='Yes').values_list().count()
    terminated_female=emlploy_data.objects.filter(Gender='Female',Attrition='Yes').values_list().count()

    #attrition based Terminations and Activations by Yearatcompany
    year_at_company=list(emlploy_data.objects.all().annotate(YearsAtCompany1=Cast('YearsAtCompany',IntegerField())).order_by('YearsAtCompany1').values_list("YearsAtCompany1",flat=True).distinct())
    print(year_at_company)
    active_year_values=[]
    terminated_year_values=[]
    for i in year_at_company:
        active_year_values.append(emlploy_data.objects.filter(YearsAtCompany=i,Attrition='No').values_list().count())
        terminated_year_values.append(emlploy_data.objects.filter(YearsAtCompany=i,Attrition='Yes').values_list().count())
    print(active_year_values)
        
    # #attrition based by age group
    # under19=list(emlploy_data.objects.all())



    

    return render(request,'dashboard.html',{"total_headcount":total_headcount,
                                            "active_employ":active_employ,
                                            "terminated_employ":terminated_employ,
                                            "attrition_rate":attrition_rate,
                                            "dept_labels":list(dept_labels),
                                            "dept_values":dept_values,
                                            "active_males":active_males,
                                            "active_females":active_females,
                                            "terminated_male":terminated_male,
                                            "terminated_female":terminated_female,
                                            "year_at_company":year_at_company,
                                            "active_year_values":active_year_values,
                                            "terminated_year_values":terminated_year_values})


def AI_prediction (request):
    if request.method=='POST':
        employ_data_df=pd.DataFrame(request.POST.dict(),index=[0])

        DistanceFromHome=request.POST.get('DistanceFromHome')
        Age=request.POST.get('Age')
        DailyRate=request.POST.get('DailyRate')
        EnvironmentSatisfaction=request.POST.get('EnvironmentSatisfaction')
        JobInvolvement=request.POST.get('JobInvolvement')
        JobLevel=request.POST.get('JobLevel')
        JobRole=request.POST.get('JobRole')
        JobSatisfaction=request.POST.get('JobSatisfaction')
        MaritalStatus=request.POST.get('MaritalStatus')
        MonthlyIncome=request.POST.get('MonthlyIncome')
        NumCompaniesWorked=request.POST.get('NumCompaniesWorked')
        OverTime=request.POST.get('OverTime')
        RelationshipSatisfaction=request.POST.get('RelationshipSatisfaction')
        StockOptionLevel=request.POST.get('StockOptionLevel')
        TotalWorkingYears=request.POST.get('TotalWorkingYears')
        TrainingTimesLastYear=request.POST.get('TrainingTimesLastYear')
        WorkLifeBalance=request.POST.get('WorkLifeBalance')
        YearsAtCompany=request.POST.get('YearsAtCompany')
        YearsInCurrentRole=request.POST.get('YearsInCurrentRole')
        YearsWithCurrManager=request.POST.get('YearsWithCurrManager')

#create  data frame 
        employ_data_df["DistanceFromHome"]=DistanceFromHome
        employ_data_df["Age"]=Age
        employ_data_df["DailyRate"]=DailyRate
        employ_data_df["EnvironmentSatisfaction"]=EnvironmentSatisfaction
        employ_data_df["JobInvolvement"]=JobInvolvement
        employ_data_df["JobLevel"]=JobLevel
        employ_data_df["JobRole"]=JobRole
        employ_data_df["JobSatisfaction"]=JobSatisfaction
        employ_data_df["MaritalStatus"]=MaritalStatus
        employ_data_df["MonthlyIncome"]=MonthlyIncome
        employ_data_df["NumCompaniesWorked"]=NumCompaniesWorked
        employ_data_df["OverTime"]=OverTime
        employ_data_df["RelationshipSatisfaction"]=RelationshipSatisfaction
        employ_data_df["StockOptionLevel"]=StockOptionLevel
        employ_data_df["TotalWorkingYears"]=TotalWorkingYears
        employ_data_df["TrainingTimesLastYear"]=TrainingTimesLastYear
        employ_data_df["WorkLifeBalance"]=WorkLifeBalance
        employ_data_df["YearsAtCompany"]=YearsAtCompany
        employ_data_df["YearsInCurrentRole"]=YearsInCurrentRole
        employ_data_df["YearsWithCurrManager"]=YearsWithCurrManager

        

    #trasnform data
        employ_data_df["DistanceFromHome"] = pt_distance.transform(employ_data_df[["DistanceFromHome"]])
        employ_data_df["MonthlyIncome"] = pt_income.transform(employ_data_df[["MonthlyIncome"]])
        employ_data_df["NumCompaniesWorked"] = pt_companies.transform(employ_data_df[["NumCompaniesWorked"]])
        employ_data_df["TotalWorkingYears"] = pt_total_years.transform(employ_data_df[["TotalWorkingYears"]])
        employ_data_df["YearsAtCompany"] = pt_company_years.transform(employ_data_df[["YearsAtCompany"]])


        ##jobrole

        def changing_jobrole(x):
            if 'Manager' in x or 'manager' in x:
                    return 0
            elif 'director'  in x or 'Director' in x:
                return 1 
            elif 'representative'  in x or 'Representative' in x:
                return 2
            else:
                return 3 


        employ_data_df["JobRole"]=employ_data_df["JobRole"].apply(changing_jobrole)

        ##OverTime

        def changing_OverTime(x):
            if 'yes' in x or 'Yes' in x :
                    return 0
            else:
                return 1


        employ_data_df["OverTime"]=employ_data_df["OverTime"].apply(changing_OverTime)

        #one hot encoding
        employ_data_df = preprocessor.transform(employ_data_df)

        #prediction
        Attrition=pipeline_d.predict(employ_data_df)[0]
        if Attrition == 1:
            Attrition = "Low Attrition risk"
        else:
            Attrition = "High Attrition risk"

        # prediction probability
        probability = pipeline_d.predict_proba(employ_data_df)[0].max() * 100

        #to enter the predicted data in the history table 
        employ_data_hist.objects.create(Age=Age,
                                        JobRole=JobRole,
                                        MonthlyIncome=MonthlyIncome,
                                        OverTime=OverTime,
                                        Attrition=Attrition,
                                        probability=probability)

        employ_data_hist.objects.filter(id__in=[1, 2, 3]).delete()

        
                                                

        return render(request,'AI_prediction.html',{'Attrition':Attrition})

    return render(request,'AI_prediction.html')


def employyes(request):
    #base query
    employees = emlploy_data.objects.all().order_by('-id')
    status_filter=request.GET.get("status","")
    search_query=request.GET.get("search","")
    if status_filter:
        employees=emlploy_data.objects.all().filter(Attrition=status_filter)
    if search_query:
        employees=emlploy_data.objects.all().filter(JobRole=search_query)


    paginator = Paginator(employees, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'employyes.html', {'page_obj': page_obj,
                                              'employees':employees,
                                              'status_filter':status_filter,
                                              'search_query':search_query})
    

def Dep_trends (request):
    department_name=list(emlploy_data.objects.all().values_list("JobRole",flat=True).distinct())
    # year_filter=request.GET.get("year","")

    # year = list(emlploy_data.objects.values_list("YearsAtCompany",flat=True).distinct())

    # employees = emlploy_data.objects.all()

    # if year_filter:
    #     employees=employees.objects.all().filter(YearsAtCompany=year_filter)
    # print(employees)

    total_employees=[]
    active_count=[]
    terminated_count=[]
    attrition_rate=[]
    for i in department_name:
        active_count.append(emlploy_data.objects.all().filter(JobRole=i,Attrition='No').values_list().count())
        terminated_count.append(emlploy_data.objects.all().filter(JobRole=i,Attrition='Yes').values_list().count())
        total_employees.append(emlploy_data.objects.all().filter(JobRole=i).values_list().count())

    for x,y in zip(terminated_count,total_employees):
        attrition_rate.append(round((x/y)*100,2))

    #saving in trends becausa in jinja there is loop formed called trends and that why we have to send trend's list
    trends = []
    for i in range(len(department_name)):
        trends.append({
            'department_name': department_name[i],
            'total_employees': total_employees[i],
            'active_count': active_count[i],
            'terminated_count': terminated_count[i],
            'attrition_rate': attrition_rate[i]
        })
    
    return render(request,'Dep_trends.html',{'trends': trends,
                                             })




def employ_data_insert(request):
    data_csv=pd.read_csv("E:\data\employe_model\ml_model\employdata.csv")
    for i,j in data_csv.iterrows():
        emlploy_data.objects.create(Age=j['Age'],
                                    Attrition=j['Attrition'],
                                    DailyRate=j['DailyRate'],
                                    DistanceFromHome=j['DistanceFromHome'],
                                    EmployeeNumber=j['EmployeeNumber'],
                                    EnvironmentSatisfaction=j['EnvironmentSatisfaction'],
                                    JobInvolvement=j['JobInvolvement'],
                                    JobLevel=j['JobLevel'],
                                    JobRole=j['JobRole'],
                                    JobSatisfaction=j['JobSatisfaction'],
                                    MaritalStatus=j['MaritalStatus'],
                                    MonthlyIncome=j['MonthlyIncome'],
                                    Over18=j['Over18'],
                                    OverTime=j['OverTime'],
                                    RelationshipSatisfaction=j['RelationshipSatisfaction'],
                                    StandardHours=j['StandardHours'],
                                    StockOptionLevel=j['StockOptionLevel'],
                                    TotalWorkingYears=j['TotalWorkingYears'],
                                    TrainingTimesLastYear=j['TrainingTimesLastYear'],
                                    WorkLifeBalance=j['WorkLifeBalance'],
                                    YearsAtCompany=j['YearsAtCompany'],
                                    YearsInCurrentRole=j['YearsInCurrentRole'],
                                    YearsWithCurrManager=j['YearsWithCurrManager'],
                                    Department=j['Department'],
                                    EducationField=j['EducationField'],
                                    Gender=j['Gender'],
                                    BusinessTravel=j['BusinessTravel'],
                                    NumCompaniesWorked=j['NumCompaniesWorked'],
                                    YearsSinceLastPromotion=j['YearsSinceLastPromotion'],
                                    Education=j['Education'],
                                    PercentSalaryHike=j['PercentSalaryHike'],
                                    MonthlyRate=j['MonthlyRate'],
                                    HourlyRate=j['HourlyRate'],
                                    PerformanceRating=j['PerformanceRating'])

    print(request.POST)
    return JsonResponse({"message":"employy data inserted successfully"})


def employ_data_delete(request):
    emlploy_data.objects.all().delete()

    return JsonResponse({"message":"employy data deleted successfully"})

@csrf_exempt
def sending_response(request):
    print(request.POST)
    return JsonResponse({"message":"successful"})







