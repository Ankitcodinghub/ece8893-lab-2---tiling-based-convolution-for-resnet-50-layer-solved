# ece8893-lab-2---tiling-based-convolution-for-resnet-50-layer-solved
**TO GET THIS SOLUTION VISIT:** [ECE8893 Lab 2 – Tiling-based Convolution for ResNet-50 Layer Solved](https://www.ankitcodinghub.com/product/ece8893-lab-2-tiling-based-convolution-for-resnet-50-layer-with-hd-input-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;113900&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;ECE8893 Lab 2 - Tiling-based Convolution for ResNet-50 Layer  Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
Motivation

Suppose that you are given the following image and asked to draw tight-fitting boxes around various objects in the image. These could be cars, buses, pedestrians, etc. Furthermore, you have to classify each object with a different color. Seems quite trivial, right?

Here’s the final output obtained from a multi-object tracking model called QDTrack, which is the current state-of-the-art in computer vision research for this task.

Are you ready?

Convolution for Object Detection

At the core of almost every object detection model is a convolution neural network (CNN) such as VGG-16, ResNet50, Xception, YOLO, MobileNet, etc. These are essentailly feature extraction networks. They “look” at images and extract salient features such as edges, shapes, and so on.

Introduced in 2015, ResNet-50 is (still) one of the most popular CNN architectures and forms the backbone of many multi-object detection models, including QDTrack. Let’s consider the first convolution layer of ResNet-50 which involves a 7 x 7 convolution. After passing the above input image through this layer, we get 64 two-dimensional “feature maps” as shown below. As you can observe, each layer extracts different features from the input image.

We are interested in implementing this convolution layer of ResNet-50 with the above HD input image. Suppose that the input image (feature map) is described by a 3D tensor X(ID, IH, IW). We use a filter W with 64 kernels, each of dimensions (ID, KH, KW) where KH and KW are kernel window heights and widths. The resulting output feature map can be described by another 3D tensor Y(OD, OH, OW). The output feature map dimesions are a function of the stride S and padding size P chosen. For the first layer of ResNet-50, the values of these parameters are described in the table below.

| Layer Parameters | Layer Values | | —————- | —————– | | Kernel Size (KH x KW) | 7 x 7 | | Filter Size (OD) | 64 | | Input Feature Map (ID, IH, IW) | (3, 736, 1280) | | Stride (S) | 2 | | Padding (P) | 3 |

| Output Feature Map (OD, OH, OW) | (64, 368, 640) |

If these values do not make any sense to you in the first glance, don’t worry. You can learn these concepts easily through this well-written CNN cheatsheet.

This lab will be split into following parts: – Part A: C Model creation for functional correctness (30 points) – Part B: Unoptimized but synthesizable tiled-convolution in HLS (30 points) – Part C: Optimized (and synthesizable) tiled-convolution in HLS (40 points)

Part A: Implementation of C Model for Convolution (30 points)

Before we jump into the HLS implementation and start to play around with those pragmas that we love, we need to have a functionally correct implementation prepared to assist us with debugging. Without a reference C model, it is incredibly painful to debug the tiling-based convolution that you will implement after this.

In this part of the lab, you have to write a C/C++ program that performs 7 x 7 convolution. You are provided with a testbench which reads the input image, layer parameters, as well the reference output feature map and performs a point-to-comparison with the obtained feature map. Mean Squared Error (MSE) is used as the metric for evaluation.

bin contains the reference binary files

sim.cpp is the testbench that checks for functional correctness

model_conv.cpp is the source code file you need to update to implement 7 x 7 convolution

Note: This part of the lab is simulation-only. You do not have to run Vitis synthesis!

Reference Material: [Stanford CNN Cheatsheet][https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks]

How to Run the Code

Just like in Lab 1, simply use the provided Makefile. Run make and then execute the output (if successfully generated) with csim.out.

What to Submit

1. model_conv.cpp that passes the functional test.

2. A brief report (preferably in PDF format) describing:

3. What is the MSE obtained?

4. How did you handle the non-unit stride and border pixels?

Part B: Tiling-based Unoptimized Convolution (30 points)

To address the challenges that come with scalability (such as large input size), a typical approach for hardware implementation is to split the input image into tiles and run convolution on each tile. The overall latency is then the latency of individual tile times the number of tiles in the image.

In this part of the lab, you have to implement an unoptimized but synthesizable code that performs tiling-based convolution of the first 7 x 7 layer of ResNet-50 with HD input image. Similar to Part A, you are provided with a test bench and we will continue to use MSE as the evaluation metric. bin/ contains the reference binary files – sim.cpp is the testbench that checks for functional correctness of your tiling-based convolution. tiled_conv.cpp is the top-level design that you will synthesize after implementing the convolution operation. – utils.cpp has utility functions to assist you with loading and storing of data. You can modify these functions as required or add any new ones for your design. – conv_7x7.cpp is the convolution engine that performs 7 x 7 convolution for a single tile.

Please read this article to implement tiling-based convolution. Be mindful that this time the stride is 2 and the padding is 3 (on each side).

While loading a tile, you would need to load some additional “features” of adjacent tiles. This is defined by MARGIN in the conv.h header file which you need to update. DO NOT change any other values in conv.h (as this file is not required for submission).

Note: You DO NOT need to add optimization pragmas in this part of the lab. The goal is to implement a trivial tiling-based convolution layer implementation that is functionally correct and establish a baseline.

Hint: Following a calcluation method similar to the one described in the above article, you will observe that the 7 x 7 layer we are dealing with comprises of ~2.21 billion MAC operations. At 100 MHz (10ns) clock, this translates to ~22.1 seconds. Your unoptimized (trivial) synthesis latency should be close to this value.

Caution: Do not use any float variables while implementing your design. Since there is a type-casting of data type from floating-point to fixedpoint for synthesis, you should always use fm_t or wt_t.

Reference Material: Tiling-based Convolution

How to Run the Code for Simulation and Synthesis

Use make to run simulation test using floating-point values.

Use make debug to run simulation and print floating-point values for debugging.

Use make hls_sim to run simulation using fixed-point values. (You will see the MSE increases!)

To run synthesis, use make synth. Once synthesis completes, you can find the report file here:

proj/solution1/syn/report/csynth.rpt What to Submit

1. PartB.tar.gz that contains the following:

2. utils.h

3. utils.cpp

4. conv_7x7.cpp

5. tiled_conv.cpp

6. unoptimized_csynth.rpt (Please rename PartB/proj/solution_1/syn/report/csynth.rpt to unoptimized_csynth.rpt)

7. A brief report (preferably in PDF format) describing:

8. How many marginal features are needed to implement tiling?

9. What is the MSE obtained in floating-point simulation?

10. What is the MSE obtained in fixed-point simulation?

11. What is the unoptimized latency and resource utilization obtained?

12. What is the communication overhead? (Communication overhead is the ratio of load/store latency and the computation latency)

Part C: Tiling-based Optimized Convolution (40 points + up to 10 points)

Now for the most fun (and challenging) part of this lab (and course). Take your Part B code and apply all your favorite pragmas and other techniques to optimize your convolution implementation.

Your target in this part of the lab is to achieve a latency of 750 ms or lower without exceeding 100% resource utilization for the Pynq-Z2 board. You are also required to estimate how much of a speedup your optimized 7 x 7 tiling convolution code should hit to meet this overall speedup (Amdahlâ€™s law in action!).

Furthermore, you need to estimate the communication overhead again and describe how it has changed compared to that of trivial implementation.

What to Submit

1. PartC.tar.gz that contains the following:

2. utils.h

3. utils.cpp

4. conv_7x7.cpp

5. tiled_conv.cpp

6. optimized_csynth.rpt (Please rename PartC/proj/solution_1/syn/report/csynth.rpt to optimized_csynth.rpt)

7. A brief report (preferably in PDF format) describing:

8. What is the optimized latency and resource utilization obtained?

9. What are the main techniques you adopted?

10. What is the new communication overhead and how does it compare to value obtained in Part B?

Note: Please combine your Part A, B, and C (and D) reports in a single file and submit Lab2_Report_&lt;Name&gt;.pdf. There is no template to follow, however, you are expected to write your report like a research paper.

Need for Speed…-Up Contest (chance to grab 10 extra points!)

Perform design space exploration (DSE) and achieve the best latency while not exceeding resource utilization. We will maintain a leaderboard! – If your latency is less than 500 ms or among the Top 10 of the class, you get 5 extra points. – If your latency is among the Top 5 of the class, you get 10 extra points (in total)!

Part D: HLS Streaming Exploration (up to 10 extra points!)

Implement the tiling-based convolution using streaming. You are free to change any file in this part of the lab as it will be evaluated separately.

What to Submit

1. PartD.tar.gz that containing all the files and a README.md to run simulation and synthesis (if different from current mehtod)

2. A brief report describing:

3. The latency obtained and its comparison with PartC latency

4. The overall resource utilization change

5. The challenges you faced or any observations made

Submit your analysis in the same report file.

Note: You CANNOT report your latency obtained using streaming for the latency contest.

Grading Rubric

Part A.1

simulationTestPass → +25 points

Part A.2

Missing or incomplete information → -2 points for each question

Part B

simulationTestPass â†’ +15 points

Code is synthesizable (Vitis run completes without errors) â†’ +5 points

Missing or incomplete or inconsistent information â†’ -2 points for each question

Part C

if(test pass)

if(latency ≤ 750 ms), +40 points else if (750 ms &lt; latency &lt; 1 s), +35 points else if (1 s &lt; latency &lt; 2 s), +20 points else, +10 points

for resource in [BRAM, DSP, FF, LUT]: -10 points if utilization &gt; 100%

If synthesis fails or times out (10 minutes), -10 points.

In report, any missing or incomplete or inconsistent information â†’ -2 points for each question
