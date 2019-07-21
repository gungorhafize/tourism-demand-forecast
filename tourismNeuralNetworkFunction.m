function [y1] = tourismNeuralNetworkFunction(x1)

% [y1] = tourismNeuralNetworkFunction(x1) takes these arguments:
%   x = 5xQ matrix, input #1
% and returns:
%   y = 1xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0;0;0;0;0];
x1_step1.gain = [2;2;2;2;2];
x1_step1.ymin = -1;

% Layer 1
b1 = [2.5291123180274058;-0.94209250315934578;0.47343447960103385;-1.1901554663049969;-0.62382694609071054;-0.94375553692024583;-1.779054313857195;1.6665085928560444];
IW1_1 = [-0.25410146043430082 -0.78394881798601068 1.1956341447662524 0.68422184644207062 0.55697796206269701;1.1094671349526055 0.88205538145022899 2.3376108272644851 0.27263765031954745 0.67059068202544547;-1.6591199371514946 -0.328425625217729 0.83950470123873266 0.94975668023324133 0.041787554742127808;1.7312425388757202 -1.234426982423912 -0.27845071690265294 0.21536791627968702 1.138331596933196;-1.7603098457757005 0.74302947612510761 -1.2319209195134604 -0.70480054578335039 0.56224942282094625;-0.39784886706618838 -0.93774992728509021 0.76180342319949679 0.10235318611781685 2.2050008914037869;-0.47217128617695259 0.65522766559801982 -0.40445906360459871 1.5920915486331799 -0.6476717659919986;1.1418875354450277 1.1205608288303468 -0.19603400216616051 0.49623113359431359 1.6398158736831323];

% Layer 2
b2 = 0.70308107463930958;
LW2_1 = [-0.577969849374244 0.47865558526395324 -0.25262568567735072 -0.26023855037933435 0.050999464181403165 0.24892951025582899 -0.097217392838441996 0.42841627758006057];

% Output 1
y1_step1.ymin = -1;
y1_step1.gain = 2;
y1_step1.xoffset = 0;

% ===== SIMULATION ========

% Dimensions
Q = size(x1,2); % samples

% Input 1
xp1 = mapminmax_apply(x1,x1_step1);

% Layer 1
a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*xp1);

% Layer 2
a2 = repmat(b2,1,Q) + LW2_1*a1;

% Output 1
y1 = mapminmax_reverse(a2,y1_step1);
end

% ===== MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings)
y = bsxfun(@minus,x,settings.xoffset);
y = bsxfun(@times,y,settings.gain);
y = bsxfun(@plus,y,settings.ymin);
end

% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n,~)
a = 2 ./ (1 + exp(-2*n)) - 1;
end

% Map Minimum and Maximum Output Reverse-Processing Function
function x = mapminmax_reverse(y,settings)
x = bsxfun(@minus,y,settings.ymin);
x = bsxfun(@rdivide,x,settings.gain);
x = bsxfun(@plus,x,settings.xoffset);
end
