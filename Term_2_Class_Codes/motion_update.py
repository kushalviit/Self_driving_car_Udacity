def motion_updte(mean1,var1,mean2,var2):
    new_mean=mean1+mean2
    new_var=var1+var2
    return [new_mean,new_var]

print(motion_updte(8.,4.,10.,6.))
