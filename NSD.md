# Natural Scenes Dataset
## 数据集概况
fMRI是功能性磁共振成像的缩写，是一种无创非放射性观察大脑活动的技术。它通过测量大脑区域血液流量的变化来推断大脑活动的影像技术，主要的原理是血氧水平依赖（blood oxygenation level dependent，BOLD）。fMRI已被广泛用于神经科学及认知科学领域的研究。

### nsddata
这是包含基本数据文件的主目录，包括(但不限于)解剖数据、prf和floc实验结果、行为数据、FreeSurfer主题目录和roi。

### nsddata_betas
这个非常大的文件夹包含NSD实验的估计fMRI单次试验反应(“beta”)以及相关结果(例如噪声上限估计)。beta有多个版本(例如，betas_assumehrf (b1)， betas_fithrf (b2)， betas_fithrf_GLMdenoise_RR (b3))。此外，beta可以在不同的空间(例如1.8毫米体积(func1pt8mm)、1毫米体积(func1mm)、主体原生表面(native - surface)、f平均值、MNI)中制备和使用。

> 单次试验贝塔（the single-trial betas）是指在fMRI数据中，每个试验的血氧水平响应。在fMRI数据分析中，我们通常会将每个试验的血氧水平响应与设计矩阵中的回归器相乘，得到一个beta值，这个beta值就是单次试验贝塔。（详细见**关键词**章节）
> 
> 具体来说，我们可以将每个试验的血氧水平响应与设计矩阵中的回归器相乘，得到一个beta值。这个beta值就是单次试验贝塔。这个beta值可以用来计算出每个脑区在每个试验中的活动。

> 不同版本的单次试验贝塔可能是由不同的分析软件或分析方法生成的。例如，betas_assumehrf (b1)是由SPM软件生成的，betas_fithrf (b2)是由FSL软件生成的，betas_fithrf_GLMdenoise_RR (b3)是由GLMdenoise软件生成的。
> - SPM（Statistical Parametric Mapping）是一种常用的fMRI数据分析软件，它可以用于对fMRI数据进行预处理、统计分析等。
> - FSL（FMRIB Software Library）是另一种常用的fMRI数据分析软件，它也可以用于对fMRI数据进行预处理、统计分析等。
> - GLMdenoise是一种基于线性模型的去噪方法，它可以用于去除fMRI数据中的噪声，提高信噪比。

> 这些空间代表了不同的解剖结构或分辨率。例如，1.8毫米体积(func1pt8mm)表示每个体素的大小为1.8毫米，而1毫米体积(func1mm)表示每个体素的大小为1毫米。主体原生表面(native - surface)表示大脑皮层的表面，而MNI则是一种标准空间。

### nsddata_stimuli
这包含了在NSD实验中使用的彩色自然场景图像。

### nsddata_timeseries
这个非常大的文件夹包含预处理的fMRI时间序列数据，从中估计单次试验贝塔(the single-trial betas)。有1.8毫米和1毫米两种型号可供选择。此外，该文件夹包含与时间序列数据相关的信息，包括生理数据(脉搏和呼吸)、实验设计信息(即何时显示哪些图像)、fMRI数据预处理后的运动参数估计以及眼动追踪数据。

### nsddata_other
这包含杂项，包括(但不限于)用于运行实验的材料和原始未编辑的FreeSurfer输出。

### nsddata_diffusion
这包含了分析扩散数据的导数。注意:我们目前正在准备扩散衍生文件的最终版本，并将在几周内提供。

### nsddata_rawdata
这包含了BIDS格式的原始数据。

> BIDS（Brain Imaging Data Structure）是一种用于存储和共享神经影像数据的标准格式，它是一种基于文件夹的格式，其中包含了元数据和原始数据。BIDS格式的数据可以使得数据共享更加方便，同时也可以使得数据的重复使用更加容易。

## Functional data (pRF, fLoc)
这包括在初始7T prffloc扫描会话中进行的pRF和fLoc实验的分析结果。[FMRI在1.5T,3T和7T的区别到底有多大？](https://www.zhihu.com/question/321238415)

## Functional data (NSD)
这包括NSD实验的GLM结果。NSD数据的主要GLM分析的目标是估计每个体素的单次试验β (BOLD响应幅度)。

> 在fMRI中，GLM是一种一般线性模型（general linear model）的方法，它将数据视为模型函数（预测器）和噪声（误差）的线性组合。在一阶分析中，GLM将任务开始时间/持续时间与HRF做卷积，从而量化各个条件下BOLD信号值。

### File format issues for betas
包含NSD测试版的文件非常大。所制备的betas的单位是信号变化的百分比。但是，对于我们准备的一些NSD数据文件，我们将**beta值**乘以300，并将其转换为int16格式，以减少空间使用。

**加载beta文件后，应立即将值转换为十进制格式(例如单或双)并除以300，将其转换回百分比信号变化。**（重点！！！重点）（使用时注意）

> For volume-based format of the betas, two versions have been prepared:
> - NIFTI (.nii.gz). These data are in int16 format. A liberal brain mask has been applied such that non-brain voxels have been zeroed-out in order to save disk space. The .gz indicates that the files are compressed (to save disk space). The advantage of .nii.gz format is that it is standard and easy-to-use, but the disadvantage is that the files must be uncompressed when loading and must be completely loaded into memory.
> - HDF5 (.hdf5). These data are in int16 format. '/betas' is X voxels x Y voxels x Z voxels x 750 trials. A liberal brain mask has been applied such that non-brain voxels have been zeroed-out. This file is in HDF5 format (with a specific chunk size of [1 1 1 750]) in order to enable very fast random access to small parts of the data file. A disadvantage of this format is that the file is uncompressed and therefore large in size.

### Results of a simple ON-OFF GLM
除了单次试验GLM外，还用简单的ON-OFF GLM（a simple ON-OFF GLM）对NSD进行了分析，以得出一些有用的量。

##### nsddata/ppdata/subjAA/func*/onoffbeta_sessionBB.nii.gz
这是一个简单的GLM模型在sessionBB中获得的beta值(以信号变化单位的百分比为单位)，该模型用一个简单的ON-OFF预测器(一个条件，标准的HRF(血氧水平依赖性, hemodynamic response function, HRF))描述了与实验相关的方差。
![onoffbeta_sessionBB.nii.gz](https://slite.com/api/files/3iLKxQWQ3X/image.png)

##### nsddata/ppdata/subjAA/func*/onoffbeta.nii.gz
这是所有会话中onoffbeta的平均值(使用nanmean.m)。

##### nsddata/ppdata/subjAA/func*/R2_sessionBB.nii.gz
这是sessionBB的简单ON-OFF GLM模型的**逐体素方差解释**(0-100)。

> 体素方差解释是指模型对数据的拟合程度，它可以用来评估模型的质量。Voxel-wise variance explained是一个衡量模型在每个体素上解释数据方差的指标。它是模型解释的方差与每个体素总方差之比。Voxel-wise variance explained值越高，模型在该体素上拟合数据的效果越好。

![R2_sessionBB.nii.gz](https://slite.com/api/files/2BsrY1AbSB/image.png)

##### nsddata/ppdata/subjAA/func*/R2.nii.gz
这是所有会话的体素方差(使用nanmean.m)。

##### nsddata/freesurfer/subjAA/label/[lh,rh].R2.mgz
？这是干啥的？没有说明？应该不是体素方差

## 关键词
### 体素
体素是体积元素（Volume Pixel）的简称，一张3D医学图像可以看成是由若干个体素构成的，体素是一张3D医疗图像在空间上的最小单元（功能性磁共振成像（fMRI）的基本单元）。[from "医学图像预处理--重采样"](blog.csdn.net/winner19990120/article/details/121605297)

### pRF 群感受野
> 神经元的感受野 (receptive field, RF) 是视觉系统信息处理的基本结构和功能单元。它是指一个神经元在视野里起反应的区域。要了解神经元的加工过程，首先就是要了解神经元的感受野，进而才能明确神经元所接收到的外界物理信息。神经元的感受野一般是通过动物电生理实验获得。而在人类脑成像研究中，由于功能性磁共振成像 (fMRI) 的分辨率在毫米量级，一个体素 (voxel，fMRI 的基本单元 ) 中就包含着成千上万个神经元。体素接收到的外界物理信息就不能简单地用单个神经元的感受野来进行推理，必须对群体的感受野进行刻画，由此产生了体素的群感受野 (population receptive field, pRF) 这个名词。换句话说，pRF 是指体素内的一群神经元在视野里起反应的区域。而从计算角度来说，体素的 pRF 是指一个 fMRI 体素内神经元群体对刺激的累计反应的一个量化模型。在视觉研究领域，体素的 pRF 模型能够很好地被用于解释和预测单个体素对刺激位于不同视野位置时的反应。
> 
> pRF 模型目前主要通过 fMRI 实验获得，这种技术被叫做基于 fMRI 的 pRF 技术。基于 fMRI 的 pRF 技术最近几年发展非常迅速。它在技术上的最大优势在于研究者可以通过这种非侵入式的技术获 得人类大脑里每个体素的 pRF 信息，这些 pRF 信息包括 ：pRF 的位置和 pRF 的大小。
>
> [群感受野技术在感知觉的脑机制研究领域的应用](http://psy.pku.edu.cn/docs/20190513095521212518.pdf)

另外，在CNNs中也有感受野(Receptive Field)概念。

### 单次试验贝塔（the single-trial betas）
单次试验贝塔（the single-trial betas）是指fMRI时间序列数据中每个试验的血氧水平响应与设计矩阵中的回归器相乘得到的beta值。它可以用以下公式计算：

$$\beta_{i,j} = (X^TX)^{-1}X^TY_{i,j}$$

其中，$i$表示第$i$个试验，$j$表示第$j$个体素，$X$是设计矩阵，$Y_{i,j}$是第$i$个试验中第$j$个体素的时间序列数据。

设计矩阵中的回归器是指用于描述fMRI时间序列数据中每个试验的血氧水平响应的一组变量。这些变量通常是由实验设计决定的，例如，如果实验中有两个条件（A和B），那么设计矩阵中可能包含两个回归器，分别对应于条件A和条件B。

常用的回归器包括：
- Boxcar函数：在试验期间为1，否则为0。
- Gamma函数：在试验期间呈现出更复杂的形状，可以更好地拟合血氧水平响应。
- Derivatives of Gamma函数：Gamma函数的导数，可以更好地拟合血氧水平响应的上升和下降阶段。
