## Conditional Random Field(CRF) 알고리즘

CRF 알고리즘은 영상 처리 중 이미지 Segmentation의 결과를 후처리할 때 많이 쓰이는 알고리즘입니다.

모든 정보는 해당 속성의 고유 특징와 주위 속성간의 관계 특징으로 

Vessel Segmentation의 경우를 생각해보자. 
특정 픽셀이 Vessel인지 아닌지 결정하는 건, 크게 2가지를 나눠서 보아야 한다. 
첫 번째로는 그 특정 픽셀만의 정보를 파악하여, 그 픽셀이 Vessel인지 아닌지를 파악하고, 
두 번째로는 그 특정 픽셀 주위의 픽셀들을 파악하여, 주위 픽셀들이 과연 Vessel인지 아닌지를 파악해보는 것이다.
즉 그 픽셀의 정보와 주위 픽셀 간 관계의 정보를 조합하여, Vessel인지 아닌지를 정한다면, 좀 더 정확한 모델이 되지 않겠냐는 것이 CRF 알고리즘의 기본 골자이다.

**Energy Function of the dense CRF**
$$
E(x) = \sum_{i} \phi_i(x_i | I) + \sum_{i, j} \psi_{i,j}(x_i,y_i | I) \\
x \rightarrow 픽셀의 label,\ I \rightarrow 픽셀의\ 이미지정보 \\ 
 \\\ \\\ \\\
목표 : Energy\ Function(E(x))을\ 최소화시키는\\ x값(픽셀의 label)들의\ 집합을\ 구해보자
$$
**1번째 항**($\sum_i \phi_i(x_i) $) : **Unary Potential**

Data Term 이라고 불리며, 각 픽셀 별 확률 정보를 의미한다.
$$
\phi_i(x_i) = - log P(x_i) \ P(x_i) : 각\ 픽셀의\ Softmax\ 값
$$
각 픽셀의 softmax 값에다가 -log를 씌어준 값이 된다.
아래는 pydensecrf로 Unary Potential을 계산해주는 식이다

````python
import tensorflow as tf
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, unary_from_softmax  

# Tensorflow model Restoring
sess = tf.Session()
saver = tf.train.import_meta_graph("./model/best_accuracy.ckpt.meta")
init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
sess.run(init)
saver.restore(sess, tf.train.latest_checkpoint("./model/"))
graph = tf.get_default_graph()
softmax = graph.get_tensor_by_name("softmax:0")

# image : (Height, Width, 3)의 형태를 가진 일반적인 이미지 numpy array 
height, width, _ = image.shape
softmax = sess.run(model.softmax, feed_dict = {x:image,feed_prob:1.0})
# 채널이 제일 앞으로 와야 함. (height, width, channels) -> (channels, height, width)
transposed_softmax = softmax.transpose((2,0,1)) 

# DenseCRF 선언
nlabels = 2 # 라벨 수
dense_crf = dcrf.DenseCRF2D(width,height,nlabels) # dense_crf 선언

unary = unary_from_softmax(transposed_softmax) # softmax값으로부터, unary Potential 값으로 바꾸어 줌.
dense_crf.setUnaryEnergy(unary) # set the Unary Potential
````

**2번째 항**($ \sum_{i, j} \psi_{i,j}(x_i,y_i)$) : **Pairwise Potential**

Pairwise potential은 일반적으로 이웃 픽셀과 색상 유사도에 의해 결정되는 식이다. 
$$
\psi_{i,j}(x_i,x_j) = \omega_1 exp ( - \frac{\lVert {p_i-p_j}\rVert^2}{2\sigma_\alpha^2} -  \frac{\lVert {I_i-I_j}\rVert^2}{2\sigma_\beta^2}) + \omega_2 exp(- \frac{\lVert {p_i-p_j}\rVert^2}{2\sigma_\gamma^2})
$$
이 수식을 좀 더 뜯어 보자. $p_i, p_j$ 는 해당 픽셀 위치, $I_i, I_j$ 는 i, j의 색상 값을 의미한다.  
1번째 항은 Appearance Kernel, 2번째 항은 Smoothness kernel이라고 하는데, 1번째 항의 비중이 커질수록 Detail해지고, 2번째 항의 비중이 커질수록 Smoothing해진다. 
Energy Function에서 Unary Potential은  학습된 모델의 Softmax를 이용하기 때문에 학습이 필요없는 부분이지만, 
Pairwise Potential은 각 픽셀 간의 관계 정보를 학습해야 한다. 

````python
# pairwise Potential 선언
dense_crf.addPairwiseBilateral(sxy=(80,80), # Color-dependent Term (x,y,r,g,b) 
                           srgb=(13,13,13), # "
                           rgbim=image, # rgbim : rgb color image-array // dtype == np.uint8
                           compat=1, # label-compatibility
                           kernel=dcrf.DIAG_KERNEL, # Kernel Type 
                           normalization=dcrf.NORMALIZE_SYMMETRIC) # Normalization Type
Q = dense_crf.inference(3)
res = np.argmax(Q, axis=0).reshape((height, width))
````

### 

### Code 정리

`````python
import tensorflow as tf
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, unary_from_softmax  

# Tensorflow model Restoring
sess = tf.Session()
saver = tf.train.import_meta_graph("./model/best_accuracy.ckpt.meta")
init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
sess.run(init)
saver.restore(sess, tf.train.latest_checkpoint("./model/"))
graph = tf.get_default_graph()
softmax = graph.get_tensor_by_name("softmax:0")

# image : (Height, Width, 3)의 형태를 가진 일반적인 이미지 numpy array 
height, width, _ = image.shape
softmax = sess.run(model.softmax, feed_dict = {x:image,feed_prob:1.0})
# 채널이 제일 앞으로 와야 함. (height, width, channels) -> (channels, height, width)
transposed_softmax = softmax.transpose((2,0,1)) 

# DenseCRF 선언
nlabels = 2 # 라벨 수
dense_crf = dcrf.DenseCRF2D(width,height,nlabels) # dense_crf 선언

unary = unary_from_softmax(transposed_softmax) # softmax값으로부터, unary Potential 값으로 바꾸어 줌.
dense_crf.setUnaryEnergy(unary) # set the Unary Potential

# pairwise Potential 선언
dense_crf.addPairwiseBilateral(sxy=(80,80), # Color-dependent Term (x,y,r,g,b) 
                           srgb=(13,13,13), # "
                           rgbim=image, # rgbim : rgb color image-array // dtype == np.uint8
                           compat=1, # label-compatibility
                           kernel=dcrf.DIAG_KERNEL, # Kernel Type 
                           normalization=dcrf.NORMALIZE_SYMMETRIC) # Normalization Type

Q = dense_crf.inference(3) # Inference Step 수 : 일반적으로 3번 이상이면 비슷한 결과가 나오기 시작
res = np.argmax(Q, axis=0).reshape((height, width))
`````

