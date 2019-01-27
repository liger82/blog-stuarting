import numpy as np
class Perceptron():
    # 역치와 학습률 지정
    def __init__(self, thresholds=0.0, eta=0.01, n_iter=10):
        self.thresholds=thresholds
        self.eta = eta
        self.n_iter=n_iter
        
    # 출력값과 목표값이 달랐을 때 가중치 업데이트하는 함수    
    def fit(self, X, y):
        # 0으로 가중치를 초기화
        self.w_ = np.zeros(1+X.shape[1])
        #self.w_ = [0.3, 0.4, 0.1]
        self.errors_ = []
        
        i=0
        while True:
            i+=1
            errors = 0
            # zip을 통해 동시에 vector X,y의 개별 값들을 넘겨줄 수 있음
            for xi, target in zip(X,y):
                # 예측값(출력값)
                predicted = self.predict(xi)
                if predicted != target :
                    # 이부분은 동일하므로 한번만 계산한다
                    update = self.eta * (target - predicted)
                    # 가중치 업데이트
                    self.w_[1:] += update * xi
                    self.w_[0] += update # x0은 1이므로 곱할 필요없음
                    errors += int(update!=0.0)
            if errors ==0:
                print(i," weight : ",self.w_)
                return self
            self.errors_.append(errors)
            print(i," weight : ",self.w_)
        return self

    
    # net값 구하는 함수
    def net_input(self, X):
        # dot()은 인자가 벡터일 경우 동일 인덱스별로 곱하여 합해준다.
        # x0은 1이므로 곱하기 생략
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    # unit step function을 activation function으로 사용하여 예측값 출력
    def predict(self, X):
        # where(조건, 참일때 값, 거짓일때 값)
        return np.where(self.net_input(X) > self.thresholds,1,0)
        
    if __name__=='__main__': # and 연산
    X= np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])
    
    ppn = Perceptron(eta=0.1)
    ppn.fit(X,y)
    print(ppn.errors_)
    
    '''
    if __name__=='__main__': # or 연산
    X= np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,1])
    
    ppn = Perceptron(eta=0.1)
    ppn.fit(X,y)
    print(ppn.errors_)
    '''
