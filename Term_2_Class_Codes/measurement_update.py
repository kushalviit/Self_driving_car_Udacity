#Mean and Variance Update for prediction/motion update

def update(mean1,var1,mean2,var2):
    mean1=mean1*1.
    mean2=mean2*1.
    var1=var1*1.
    var2=var2*1.
    new_mean=(mean1*var2 + mean2*var1)/(var1+var2)
    new_var=(var1*var2)/(var1+var2)
    return [new_mean,new_var]

print(update(10.,8.,13.,2.))
