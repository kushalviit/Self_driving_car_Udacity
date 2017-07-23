#include "kalman_filter.h"
#include <cmath>
#include <cstdio>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
   x_=F_*x_;
   MatrixXd Ft=F_.transpose();
   P_=F_*P_*Ft+Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
        VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
   float px = x_(0);
   float py = x_(1);
   float vx = x_(2);
   float vy = x_(3);
   VectorXd h(3);
   float c1=px*px+py*py;
   float c2=sqrt(c1);
  
   
  if(fabs(c1) < 0.0001){
		cout << "UpdateEKF - Error - Division by Zero" << endl;
		h=z;
	}
   else if(fabs(px)<0.0001 )
   {
     h=z;
   }
   else
   {
     float c3=atan2(py,px);
     float c4=(px*vx+py*vy)/c2;
     h<<c2,c3,c4;
   }

   
  
   VectorXd y=z-h;
   
   
   while(y(1)<=(-1*M_PI))
     y(1)+=2*M_PI;

    
   while(y(1)>=(M_PI))
     y(1)-=2*M_PI;
   
   MatrixXd Ht = H_.transpose();
   MatrixXd S = H_ * P_ * Ht + R_;
   MatrixXd Si = S.inverse();
   MatrixXd PHt = P_ * Ht;
   MatrixXd K = PHt * Si;
  


   //new estimate
   x_ = x_ + (K * y);
   long x_size = x_.size();
   MatrixXd I = MatrixXd::Identity(x_size, x_size);
   P_ = (I - K * H_) * P_;
   
}
