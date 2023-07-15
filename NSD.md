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

### Results of single-trial GLM
对于单试验GLM，我们使用3种不同的GLM模型分析NSD实验的时间序列数据。这些模型的标识符是:
- **betas_assumehrf (beta version 1; b1)** - GLM in which a canonical HRF is used. （使用规范HRF的GLM）
- **betas_fithrf (beta version 2; b2)** - GLM in which the HRF is estimated for each voxel.（估计每个体素的HRF的GLM。）
- **betas_fithrf_GLMdenoise_RR (beta version 3; b3)** – GLM in which the HRF is estimated for each voxel, the GLMdenoise technique is used for denoising, and ridge regression is used to better estimate the single-trial betas.（其中估计每个体素的HRF, glm降噪技术用于去噪，ridge regression用于更好地估计单次试验贝塔。）

从这些GLMs中获得的betas的解释是，它们是每次刺激试验引起的BOLD反应幅度，相对于没有刺激时存在的基线信号水平(“灰屏”)。注意，betas是用信号变化的百分比来表示的，它是用一个体素获得的全部振幅除以在给定扫描过程中观察到的该体素的大平均强度，然后乘以100。

在主体本身的体积空间(func1mm和func1pt8mm)、主体本身的表面空间(nativesurface)以及群体空间(f平均和MNI)中都提供beta。稍后将提供有关本机表面和组空间的详细信息。

注意，为了节省磁盘空间，为func1pt8mm空间提供了'betas_assumehrf'版本，但没有为func1mm空间提供。

##### nsddata_betas/ppdata/subjAA/func*/betas_*/betas_sessionBB.[nii.gz,hdf5]
这些是单次试验的beta值(已乘以300并转换为整数格式)。beta是按时间顺序排列的。**有750个beta，因为在每个扫描会话中有750个刺激试验**(在连接所有12次运行之后)。beta与subjAA在sessionBB中获得的数据相对应。

##### nsddata_betas/ppdata/subjAA/func*/betas_*/meanbeta.nii.gz
##### nsddata_betas/ppdata/subjAA/func*/betas_*/meanbeta_sessionBB.nii.gz
对于每个会话，计算所有单次试验betas的平均值(meanbeta_sessionBB)；然后，在所有扫描过程中取平均值(meanbeta)。结果是一个体积，表示subjAA获得的逐体素平均单次试验beta。(请注意，虽然文件格式是单一的，但这些值仍然必须除以300才能返回到信号变化单位的百分比。)（一个体素，750个刺激实验的平均值）
![meanbeta.nii.gz](https://slite.com/api/files/IcvXivk6bc/image.png)

##### nsddata_betas/ppdata/subjAA/func*/betas_*/R2.nii.gz
##### nsddata_betas/ppdata/subjAA/func*/betas_*/R2_sessionBB.nii.gz
这包含GLM模型在每个会话中解释的方差(R2_sessionBB.nii.gz)，以及该数量在所有会话中的平均值(R2.nii.gz)。

请注意，“betas_assumehrf”和“betas_fithrf”模型的R2值可能不是很有用，因为这些模型非常灵活，基本上可以拟合给定时间序列中的几乎所有方差(即使时间序列没有可靠的刺激诱发反应)。相反，“betas_fithrf_GLMdenoise_RR”的R2可能是有用的，因为ridge-regression正则化确实根据每个给定体素的数据中出现的响应可靠性缩小了模型。在R2_sessionBB.nii.gz中可能存在无效体素（为`NaNs`）。对于R2.nii.gz，我们使用nanmean计算平均值。
![R2.nii.gz](https://slite.com/api/files/_fi_drYdKE/image.png)

##### nsddata_betas/ppdata/subjAA/func*/betas_*/R2run_sessionBB.nii.gz
这包含由GLM模型解释的方差，该模型为给定会话中的每次运行分别计算。

##### nsddata_betas/ppdata/subjAA/func*/betas_*/HRFindex_sessionBB.nii.gz
##### nsddata_betas/ppdata/subjAA/func*/betas_*/HRFindexrun_sessionBB.nii.gz
每个体素所选HRF的索引(1到20之间的整数)。这是对会话(HRFindexrun_sessionBB)中每次运行的估计。用于分析整个数据会话的最终HRF是通过组合跨运行的结果来确定的(HRFindex_sessionBB)。

> 在fMRI中，每个体素所选的HRF是指在进行神经影像学分析时，对于每个体素，所选择的血氧水平依赖性响应函数（HRF）的索引。这个索引通常是一个介于1到20之间的整数，用于描述神经元活动引起的血氧水平变化。
> 
> **HRF的索引值越大，表示响应函数的时间延迟越长，即神经元活动引起的血氧水平变化的响应时间越长。通常意味着神经元活动的持续时间更长，或者神经元活动的强度更弱。可能意味着神经元的活动更加缓慢或者更加微弱。这可能发生在许多情况下，例如在睡眠、休息或者放松时，神经元的活动可能会变得更加缓慢。此外，神经元的活动可能会受到药物、疾病或其他因素的影响而变得更加微弱。**

![HRFindex_sessionBB.nii.gz](https://slite.com/api/files/c0HwCMD7Qd/image.png)

##### nsddata_betas/ppdata/subjAA/func*/betas_*/FRACvalue_sessionBB.nii.gz
为每个体素选择的分数正则化水平。请注意，无效体素(例如，大脑外)被给予1的分数。

![FRACvalue_sessionBB.nii.gz](https://slite.com/api/files/LKWp7qPOTe/image.png)

### Single-trial GLM results in nativesurface format
##### nsddata_betas/ppdata/subjAA/nativesurface/betas_*/[lh,rh].betas_sessionBB.hdf5
这些文件包含了给定受试者在FreeSurfer表面空间中的贝塔值。它们被保存为`.hdf5`格式，以允许快速访问可用顶点的子集。

为了生成这些贝塔值，我们首先将1毫米的受试者原生体积贝塔值，通过立方插值的方法重新采样到受试者原生的皮层表面上（这些表面存在于3个不同的深度），然后对不同深度的贝塔值进行平均。最终得到的矩阵的维度是顶点数乘以试次数（并且按照半球分开）。

注意，贝塔值被保存在int16格式中，并且被乘以300。在某些扫描会话中，如果由于头部运动，某些空间位置移出了成像视野，那么可能会出现顶点的贝塔值全为零的情况。（在皮层表面顶点中，出现数据缺失的情况非常少；更多信息请参见nsddata/information/knowndataproblems.txt。要检测这种情况，可以简单地在每个扫描会话中检查某个顶点的所有贝塔值是否等于0。）`.hdf5`文件的‘ChunkSize’是[1 T]，其中T是总试次数；这使得加载单个顶点（或小组顶点）的所有试次数据非常快。

### Single-trial GLM results in group spaces (fsaverage, MNI)
受试者原生空间的主要优势是它们提供了NSD数据的最高分辨率版本。然而，可能会有对群体分析感兴趣的情况，这时可能需要在分析之前将NSD数据转换到群体空间。（注意，理论上，也可以先对受试者原生数据进行分析，然后在分析过程的最后将结果转换到群体空间；这样可能会得到相似但不完全相同的结果。）

群体空间的贝塔值是通过将受试者原生1毫米体积空间的贝塔值重新采样到群体空间来得到的（关于fsaverage和MNI的重新采样过程的更多细节在下面提供）。因此，群体空间的贝塔值中存在一些额外的插值（和分辨率损失）。

##### nsddata_betas/ppdata/subjAA/fsaverage/betas_*/[lh,rh].betas_sessionBB.mgh
这些文件包含了在FreeSurfer fsaverage空间中的贝塔值。为了生成这些贝塔值，我们从受试者原生表面格式开始（即将1毫米受试者原生体积空间的贝塔值，通过立方插值重新采样到受试者原生皮层表面（这些表面存在于3个不同的深度），然后对结果的贝塔值在深度上进行平均），但是我们还通过最近邻插值将它们映射到fsaverage表面。注意，贝塔值是以小数格式保存的，并且是以百分比信号变化单位（即它们没有乘以300）。在数据缺失的情况下，贝塔值可能会有NaNs。

##### nsddata_betas/ppdata/subjAA/MNI/betas_fithrf/betas_sessionBB.nii.gz
这些文件包含了在MNI空间中的贝塔值。为了生成这些贝塔值，我们将1毫米受试者原生体积空间的贝塔值，通过立方插值重新采样到MNI空间。注意，数值是以int16格式保存的，并且乘以了300。注意，对于某个扫描会话，如果某个体素的数据无效（要么是因为受试者原生体积中缺少数据，要么是因为位置在受试者原生脑掩膜之外），那么它的贝塔值将被设为全零。最后，注意，为了节省磁盘空间，我们只提供了MNI空间中的`betas_fithrf`版本的贝塔值（我们没有包括‘betas_assumehrf’和‘betas_fithrf_GLMdenoise_RR’）。

##### nsddata_betas/ppdata/subjAA/MNI/betas_fithrf/valid_sessionBB.nii.gz
这些文件对应于betas_sessionBB.nii.gz，它们表示了每个扫描会话中哪些体素包含了**有效的数据**。

### Noise ceiling


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

### BOLD 血氧水平依赖
血氧水平依赖（blood oxygenation level dependent，BOLD）

### HRF 血氧水平依赖性响应函数
HRF是血氧水平依赖性响应函数（hemodynamic response function）的缩写，它是一种用于描述神经元活动与血氧水平变化之间关系的函数。HRF通常被用于fMRI数据分析中，以便将神经元活动转换为血氧水平变化。HRF可以使用数学公式表示：

$$ HRF(t) = \frac{(t / \tau)^{a-1} e^{-t/\tau}}{\tau \Gamma(a)} $$ 

其中，$t$是时间，$\tau$是HRF的时间常数，$a$是HRF的形状参数，$\Gamma(a)$是Gamma函数。

> 时间常数是HRF的一个参数，它描述了HRF的响应时间。时间常数越小，HRF的响应时间就越短，反之亦然。形状参数是HRF的另一个参数，它描述了HRF的形状。形状参数越大，HRF的峰值就越高，反之亦然。例如，在fMRI数据分析中，通常使用的HRF具有时间常数为0.9秒和形状参数为6的Gamma函数形式。

### FreeSufer表面空间
FreeSurfer表面空间是指将fMRI信号从体积空间（即三维像素）映射到表面空间（即三维顶点）。这样可以更好地反映脑皮层结构和功能信息。

### fsaverage空间
fsaverage空间是一种基于FreeSurfer软件的群体表面空间，它是由40个正常受试者的皮层表面平均而成的。它可以用于进行群体分析或可视化。

### MNI空间
MNI空间是一种基于蒙特利尔神经学研究所（MNI）模板的群体体积空间，它是由多个正常受试者的结构MRI数据平均而成的。它可以用于进行群体分析或可视化。

### 立方插值
立方插值是一种用于重采样数据的方法，它利用相邻数据点之间的三次多项式来估计新数据点。它可以提高数据精度和平滑度。

### 群体分析
群体分析是指将多个受试者的fMRI数据进行统计比较或整合，以发现共同的脑功能模式或差异。

### 群体空间
群体空间是指将不同受试者的fMRI数据对齐到一个公共的参考空间，以便于进行群体水平的比较和统计。常用的群体空间有fsaverage和MNI等。