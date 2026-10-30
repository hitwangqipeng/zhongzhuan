function result = demo_ble_iq_cir_broadening()
%DEMO_BLE_IQ_CIR_BROADENING 对比理想与异常频域 IQ，并绘制 IFFT CIR。
%
% 使用方法：
%   result = demo_ble_iq_cir_broadening;
%
% 只使用基础 MATLAB，无需 Signal Processing / Bluetooth Toolbox。
% 建议使用 MATLAB R2020a 或更新版本，以便直接读取 UTF-8 中文注释。
%
% 生成 5 张图：
%   01 两端幅度下降；02 二次相位误差；03 未分辨近邻多径；
%   04 边缘缺点、Hann/Hamming、中间缺点、线性相位、2*pi 折返等对照；
%   05 相邻频点相位差为开口向下二次函数（即群时延呈二次变化）。
% 同时保存 PNG、FIG、IQ/CIR MAT 数据和宽度指标 CSV。
%
% 物理约定：
%   1) 每个 IQ 对应一个频点，按频率递增排列，不是单频时域 ADC 样本。
%   2) 默认演示双程合成 IQ：相位=-4*pi*f*d/c，距离=c*tau/2。
%   3) 80 个 1 MHz 等间隔频点是合成示例，不是标准 CS 有效信道表。
%      首末频点跨度为 79 MHz；DFT 孔径尺度 B=N*df=80 MHz。
%   4) 所有案例共用同一频率网格和 NFFT；缺失点在原网格上置零。
%   5) 不额外加窗，只有明确标为 Hann/Hamming 的案例使用对应权重。
%   6) 各 CIR 按自身峰值归一化，方便比较形状；原始峰高另存到指标。
%   7) 宽度为功率半高（-3.0103 dB）宽度，即幅度峰值的 1/sqrt(2)。
%      补零只增加曲线采样密度，不增加物理分辨率。
%   8) 多径例直接构造最终 IQ 中的两条等效时延分量，未模拟从单程
%      两径 H(f) 到双方 PCT 乘积 H(f)^2 所产生的全部交叉项。

%% 可修改参数
cfg.numTones = 80;
cfg.dfHz = 1e6;
cfg.fcHz = 2.44e9;
cfg.nfft = 16384;
cfg.c = 299792458;
cfg.roundTrip = true;          % true: 双程 IQ；false: 单程信道 IQ
cfg.distanceM = 8;             % 理想单径及多径中最早路径的距离
cfg.commonPhaseRad = 0.4;
cfg.gaussianSigmaFraction = 0.23;  % 高斯幅度包络 sigma/B
cfg.phaseQuadraticRad = 2.2;    % 频带边缘二次相位误差约为该值
% 以下参数描述“相邻频点残余相位差”的二次曲率，而不是残余相位本身。
% 构造 deltaEpsilon(k)=常数-C*x(k)^2，x 为相邻频点中点的归一化频率。
% 因而中心与边缘的相位差之差约为 C rad，C>0 时抛物线开口向下。
% 代码会去掉 deltaEpsilon 的离散均值，只移除一个无关的平均时延，
% 不改变其二次曲率。0.40 rad 用于清楚演示；实测时请替换为拟合值。
cfg.adjacentPhaseDiffCurvatureRad = 0.40;
cfg.pathSeparationOverResolution = 0.70;  % 两径时延差/(1/B)
cfg.linearPhaseExtraDistanceM = 1.2;
cfg.edgeMissingPerSide = 16;
cfg.centerMissingCount = 10;
cfg.phaseValidFraction = 1e-6;
cfg.plotDistanceM = [0, 18];
cfg.dbFloor = -60;
cfg.savePng = true;
cfg.saveFig = true;
cfg.outputDir = fullfile(fileparts(mfilename('fullpath')), ...
    'outputs', 'ble_iq_cir_matlab');
cfg.fontName = chooseFont();

assert(cfg.numTones >= 8 && mod(cfg.numTones, 2) == 0, ...
    'numTones 必须为至少为 8 的偶数。');
assert(cfg.nfft >= 32*cfg.numTones && mod(cfg.nfft, 2) == 0, ...
    '为稳定插值计算主峰宽度，nfft 请使用不小于 32*numTones 的偶数。');
assert(cfg.edgeMissingPerSide >= 0 && ...
    2*cfg.edgeMissingPerSide < cfg.numTones, '边缘缺点数量不合法。');
assert(cfg.centerMissingCount >= 1 && ...
    cfg.centerMissingCount < cfg.numTones, '中间缺点数量不合法。');

%% 频率网格与理想 IQ
N = cfg.numTones;
k = (0:N-1).';
fOffsetHz = (k - (N-1)/2)*cfg.dfHz;
frequencyHz = cfg.fcHz + fOffsetHz;
B = N*cfg.dfHz;
propagationFactor = 1 + double(cfg.roundTrip);
tau0 = propagationFactor*cfg.distanceM/cfg.c;
phaseIdeal = cfg.commonPhaseRad - 2*pi*fOffsetHz*tau0;
iqIdeal = exp(1j*phaseIdeal);

% 用相对中心频率建模；载频带来的常相位已吸收到各路径复增益中。
% 因此也能明确指定两条路径在中心频率处同相。
cases = struct('id', {}, 'label', {}, 'note', {}, 'iq', {}, ...
    'truePathDistanceM', {}, 'metrics', {}, 'cir', {});
cases(end+1) = makeCase('ideal', '理想单径', ...
    '平坦幅度 + 线性相位', iqIdeal, cfg.distanceM);

%% 1) 幅度包络 / 有效频带变化
ampGaussian = exp(-0.5*(fOffsetHz/(cfg.gaussianSigmaFraction*B)).^2);
cases(end+1) = makeCase('edge_gaussian', '两端幅度下降', ...
    '频点不减少，但两端权重降低，有效带宽缩小', ...
    iqIdeal.*ampGaussian, cfg.distanceM);

edgeMask = ones(N, 1);
m = cfg.edgeMissingPerSide;
if m > 0
    edgeMask(1:m) = 0;
    edgeMask(N-m+1:N) = 0;
end
cases(end+1) = makeCase('edge_missing', '边缘频点缺失', ...
    '两端频点在原始频率网格上置零，保留中间连续频带', ...
    iqIdeal.*edgeMask, cfg.distanceM);

% 显式实现对称窗，避免依赖 hann()/hamming() 所在工具箱。
winHann = 0.5 - 0.5*cos(2*pi*k/(N-1));
winHamming = 0.54 - 0.46*cos(2*pi*k/(N-1));
cases(end+1) = makeCase('hann_weight', '频域 Hann 加权', ...
    '旁瓣受抑制，同时主瓣变宽', iqIdeal.*winHann, cfg.distanceM);
cases(end+1) = makeCase('hamming_weight', '频域 Hamming 加权', ...
    '旁瓣受抑制，同时主瓣变宽', iqIdeal.*winHamming, cfg.distanceM);

%% 2) 非线性相位与线性相位对照
phaseError = cfg.phaseQuadraticRad*(2*fOffsetHz/B).^2;
cases(end+1) = makeCase('quadratic_phase', '二次相位误差', ...
    '幅度完全不变；相位斜率随频率变化，能量不能聚焦到同一时延', ...
    iqIdeal.*exp(1j*phaseError), cfg.distanceM);

% 用户关注的新模型：相邻频点的残余相位差是开口向下二次函数。
% deltaPhaseAdjacent(k)=epsilon(k+1)-epsilon(k)，所以对它累加后，
% epsilon(k) 约为三次函数。离散群时延增量为
% deltaTau_g(k)=-deltaPhaseAdjacent(k)/(2*pi*df)。
fAdjacentMidHz = 0.5*(fOffsetHz(1:end-1)+fOffsetHz(2:end));
xAdjacent = fAdjacentMidHz/max(abs(fAdjacentMidHz));
deltaPhaseAdjacent = -cfg.adjacentPhaseDiffCurvatureRad*xAdjacent.^2;
deltaPhaseAdjacent = deltaPhaseAdjacent-mean(deltaPhaseAdjacent);
phaseErrorFromAdjacent = [0; cumsum(deltaPhaseAdjacent)];
cases(end+1) = makeCase('quadratic_adjacent_phase_diff', ...
    '相邻相位差为二次函数', ...
    ['相邻频点残余相位差开口向下；等价于群时延呈二次变化，', ...
    '累积相位误差约为三次函数'], ...
    iqIdeal.*exp(1j*phaseErrorFromAdjacent), cfg.distanceM);

deltaTau = propagationFactor*cfg.linearPhaseExtraDistanceM/cfg.c;
cases(end+1) = makeCase('linear_phase', '纯线性相位变化', ...
    '对照：峰位移动，主峰宽度基本不变', ...
    iqIdeal.*exp(-1j*2*pi*fOffsetHz*deltaTau), ...
    cfg.distanceM + cfg.linearPhaseExtraDistanceM);

%% 3) 两条未分辨近邻路径
pathDeltaTau = cfg.pathSeparationOverResolution/B;
tau1 = tau0;
tau2 = tau0 + pathDeltaTau;
% 两条路径复增益均为 0.5，在中心频率同相。
% 最早路径仍在 distanceM；合成峰可能移动，也可能展宽。
iqMultipath = exp(1j*cfg.commonPhaseRad)*( ...
    0.5*exp(-1j*2*pi*fOffsetHz*tau1) + ...
    0.5*exp(-1j*2*pi*fOffsetHz*tau2));
twoDistances = cfg.c*[tau1, tau2]/propagationFactor;
cases(end+1) = makeCase('close_multipath', '未分辨近邻多径', ...
    sprintf('中心频率同相的等幅两径；时延差 %.2f ns = %.2f/B', ...
    pathDeltaTau*1e9, cfg.pathSeparationOverResolution), ...
    iqMultipath, twoDistances);

%% 对照：中间缺点和正常相位折返
centerMask = ones(N, 1);
firstMissing = floor((N-cfg.centerMissingCount)/2) + 1;
centerMask(firstMissing:firstMissing+cfg.centerMissingCount-1) = 0;
cases(end+1) = makeCase('center_missing', '中间频点缺失', ...
    '对照：主瓣不一定变宽，但旁瓣/伪峰可能抬高', ...
    iqIdeal.*centerMask, cfg.distanceM);

phaseWrapped = angle(iqIdeal);
iqFromWrappedPhase = abs(iqIdeal).*exp(1j*phaseWrapped);
cases(end+1) = makeCase('phase_wrap', '正常相位折返重建', ...
    '对照：相位在 +/-pi 折返，不改变复数 IQ，也不改变 CIR', ...
    iqFromWrappedPhase, cfg.distanceM);

%% 统一计算 CIR 与指标
numCases = numel(cases);
for n = 1:numCases
    cases(n).cir = computeCir(cases(n).iq, fOffsetHz, cfg);
    [~, phaseResidual] = phaseDiagnostics( ...
        cases(n).iq, iqIdeal, phaseIdeal, fOffsetHz, cfg);
    cases(n).metrics = measurePeak(cases(n).cir, phaseResidual);
    [~, adjacentResidual, extraGroupDelayS] = adjacentPhaseDiagnostics( ...
        cases(n).iq, iqIdeal, fOffsetHz, cfg);
    cases(n).metrics.adjacentPhaseDiffPpRad = ...
        finitePeakToPeak(adjacentResidual);
    cases(n).metrics.extraGroupDelayPpNs = ...
        finitePeakToPeak(extraGroupDelayS)*1e9;
end
base = cases(1);
id = cell(numCases, 1);
label = cell(numCases, 1);
peakDistanceM = zeros(numCases, 1);
width3dBNs = zeros(numCases, 1);
width3dBM = zeros(numCases, 1);
widthRatio = zeros(numCases, 1);
peakAmplitudeRatio = zeros(numCases, 1);
phaseResidualRmsRad = zeros(numCases, 1);
adjacentPhaseDiffPpRad = zeros(numCases, 1);
extraGroupDelayPpNs = zeros(numCases, 1);
for n = 1:numCases
    id{n} = cases(n).id;
    label{n} = cases(n).label;
    mt = cases(n).metrics;
    peakDistanceM(n) = mt.peakDistanceM;
    width3dBNs(n) = mt.width3dBNs;
    width3dBM(n) = mt.width3dBM;
    widthRatio(n) = mt.width3dBM/base.metrics.width3dBM;
    peakAmplitudeRatio(n) = mt.peakAmplitude/base.metrics.peakAmplitude;
    phaseResidualRmsRad(n) = mt.phaseResidualRmsRad;
    adjacentPhaseDiffPpRad(n) = mt.adjacentPhaseDiffPpRad;
    extraGroupDelayPpNs(n) = mt.extraGroupDelayPpNs;
end
metrics = table(id, label, peakDistanceM, width3dBNs, width3dBM, ...
    widthRatio, peakAmplitudeRatio, phaseResidualRmsRad, ...
    adjacentPhaseDiffPpRad, extraGroupDelayPpNs);

%% 数学自检：防止坐标、变换符号、宽度定义混用
assert(abs(base.metrics.peakDistanceM-cfg.distanceM) < ...
    2*cfg.c/(propagationFactor*cfg.nfft*cfg.dfHz), ...
    '理想 IQ 的峰位与设定距离不符，请检查 IFFT 符号和距离轴。');
expectedWidthNs = 0.885893/B*1e9;
assert(abs(base.metrics.width3dBNs/expectedWidthNs-1) < 0.01, ...
    '理想矩形频带的功率半高宽与 0.886/B 不符。');
linearIndex = find(strcmp({cases.id}, 'linear_phase'), 1);
assert(abs(cases(linearIndex).metrics.width3dBM/base.metrics.width3dBM-1) ...
    < 0.005, '纯线性相位不应显著改变主峰宽度。');
assert(max(abs(iqFromWrappedPhase-iqIdeal)) < 1e-12, ...
    '正常相位折返重建后，复数 IQ 应保持一致。');

%% 绘制三个主要案例，每个案例包含 6 幅对比图
if ~exist(cfg.outputDir, 'dir')
    mkdir(cfg.outputDir);
end
mainIds = {'edge_gaussian', 'quadratic_phase', 'close_multipath'};
fileNames = {'01_edge_amplitude', '02_quadratic_phase', '03_close_multipath'};
for n = 1:numel(mainIds)
    idx = find(strcmp({cases.id}, mainIds{n}), 1);
    fig = drawMainComparison(base, cases(idx), fOffsetHz, ...
        phaseIdeal, cfg);
    saveFigure(fig, fileNames{n}, cfg);
end
fig = drawControls(cases, cfg);
saveFigure(fig, '04_controls', cfg);
adjacentIndex = find(strcmp({cases.id}, ...
    'quadratic_adjacent_phase_diff'), 1);
fig = drawAdjacentPhaseDifference(base, cases(adjacentIndex), ...
    fOffsetHz, phaseIdeal, cfg);
saveFigure(fig, '05_quadratic_adjacent_phase_difference', cfg);

%% 保存：所有频域 IQ、复数 CIR、坐标和指标
result.cfg = cfg;
result.frequencyHz = frequencyHz;
result.fOffsetHz = fOffsetHz;
result.bandwidthScaleHz = B;
result.frequencySpanHz = (N-1)*cfg.dfHz;
result.propagationFactor = propagationFactor;
result.cases = cases;
result.metrics = metrics;
save(fullfile(cfg.outputDir, 'ble_iq_cir_examples.mat'), 'result');
writetable(metrics, fullfile(cfg.outputDir, 'peak_width_metrics.csv'), ...
    'Encoding', 'UTF-8');
fprintf('\n合成 IQ / CIR 对比完成。宽度采用功率半高宽。\n');
disp(metrics(:, {'label', 'peakDistanceM', 'width3dBNs', ...
    'width3dBM', 'widthRatio', 'peakAmplitudeRatio'}));
fprintf('输出目录：%s\n', cfg.outputDir);
end

function item = makeCase(id, label, note, iq, trueDistances)
item = struct('id', id, 'label', label, 'note', note, 'iq', iq(:), ...
    'truePathDistanceM', trueDistances, 'metrics', [], 'cir', []);
end

function s = computeCir(iq, fOffsetHz, cfg)
N = numel(iq);
% 注意：按频率递增的 N 个 IQ 直接传给 ifft，末尾补零。
% 不要先对这 N 个频点 ifftshift 再补零，会破坏连续频点排列。
s.tauS = (-cfg.nfft/2:cfg.nfft/2-1).'/(cfg.nfft*cfg.dfHz);
s.distanceM = cfg.c*s.tauS/(1+double(cfg.roundTrip));
hIndex = fftshift(ifft(iq(:), cfg.nfft))*cfg.nfft/N;
% 补回频率起点的相位，得到 sum[IQ_k*exp(j*2*pi*fOffset_k*tau)]/N。
% 该因子不影响 CIR 幅度或峰位，但让复数 CIR 也符合所定义的逆变换。
s.complex = hIndex.*exp(1j*2*pi*fOffsetHz(1)*s.tauS);
s.amplitude = abs(s.complex);
peak = max(s.amplitude);
assert(peak > 0, '输入 IQ 不能全部为零。');
s.normalizedAmplitude = s.amplitude/peak;
s.db = 20*log10(max(s.normalizedAmplitude, 10^(cfg.dbFloor/20)));
end

function mt = measurePeak(s, phaseResidual)
y = s.normalizedAmplitude;
[~, p] = max(y);
peakDistance = s.distanceM(p);
% 三点抛物线插值仅用于细化峰位；宽度直接用阈值交点插值。
if p > 1 && p < numel(y)
    den = y(p-1)-2*y(p)+y(p+1);
    if abs(den) > eps
        off = max(-0.5, min(0.5, 0.5*(y(p-1)-y(p+1))/den));
        peakDistance = peakDistance + off*(s.distanceM(2)-s.distanceM(1));
    end
end
threshold = 1/sqrt(2);
left = p;
right = p;
while left > 1 && y(left-1) >= threshold
    left = left-1;
end
while right < numel(y) && y(right+1) >= threshold
    right = right+1;
end
% 只测全局主峰所在的连续主瓣，避免把远端旁瓣也计入宽度。
if left == 1 || right == numel(y)
    leftTau = NaN;
    rightTau = NaN;
    leftDistance = NaN;
    rightDistance = NaN;
    warning('主峰跨越时延周期边界，未计算该峰的宽度。');
else
    leftTau = crossing(s.tauS(left-1), s.tauS(left), ...
        y(left-1), y(left), threshold);
    rightTau = crossing(s.tauS(right), s.tauS(right+1), ...
        y(right), y(right+1), threshold);
    leftDistance = crossing(s.distanceM(left-1), s.distanceM(left), ...
        y(left-1), y(left), threshold);
    rightDistance = crossing(s.distanceM(right), s.distanceM(right+1), ...
        y(right), y(right+1), threshold);
end
validResidual = phaseResidual(isfinite(phaseResidual));
mt.peakDistanceM = peakDistance;
mt.peakAmplitude = max(s.amplitude);
mt.width3dBNs = (rightTau-leftTau)*1e9;
mt.width3dBM = rightDistance-leftDistance;
mt.leftCrossingM = leftDistance;
mt.rightCrossingM = rightDistance;
mt.phaseResidualRmsRad = sqrt(mean(validResidual.^2));
end

function x = crossing(x1, x2, y1, y2, threshold)
x = x1 + (threshold-y1)*(x2-x1)/(y2-y1);
end

function [phaseUnwrapped, residual] = phaseDiagnostics( ...
    iq, iqIdeal, phaseIdeal, fOffsetHz, cfg)
% 零幅度点没有可解释相位，使用 NaN 留空。
valid = abs(iq) > cfg.phaseValidFraction*max(abs(iq));
phaseUnwrapped = NaN(size(iq));
residual = NaN(size(iq));
if nnz(valid) < 2
    return;
end

% 这是已知参考的合成实验：先移除理想相位再展开残差，避免缺口使
% 原始大斜率相位 unwrap 发生分支错误。实测数据需按信噪比稳健展开。
delta = unwrap(angle(iq(valid).*conj(iqIdeal(valid))));
phaseUnwrapped(valid) = phaseIdeal(valid) + delta;
fMHz = fOffsetHz/1e6;
fitLine = polyfit(fMHz(valid), delta, 1);
residual(valid) = delta-polyval(fitLine, fMHz(valid));
end

function [fMidHz, deltaResidual, extraGroupDelayS] = ...
    adjacentPhaseDiagnostics(iq, iqIdeal, fOffsetHz, cfg)
% 相邻频点残余相位差与由其估计的额外群时延。
% 对等间隔频点：tau_g(f_{k+1/2}) ~= -Delta phi_k/(2*pi*Delta f)。
ratio = iq(:).*conj(iqIdeal(:));
valid = abs(iq(:)) > cfg.phaseValidFraction*max(abs(iq(:)));
fMidHz = 0.5*(fOffsetHz(1:end-1)+fOffsetHz(2:end));
deltaResidual = NaN(numel(iq)-1, 1);
pairValid = valid(1:end-1) & valid(2:end);
if nnz(valid) < 2
    extraGroupDelayS = deltaResidual;
    return;
end
% 直接计算相邻复比值的相角，避免幅度缺口跨段 unwrap 污染后续频点。
% 这要求真实相邻残余相位差落在 (-pi, pi]；BLE 频点间隔下通常应先
% 去除已知粗时延，使该条件成立。
allDelta = angle(ratio(2:end).*conj(ratio(1:end-1)));
deltaResidual(pairValid) = allDelta(pairValid);
extraGroupDelayS = -deltaResidual/(2*pi*cfg.dfHz);
end

function value = finitePeakToPeak(x)
x = x(isfinite(x));
if isempty(x)
    value = NaN;
else
    value = max(x)-min(x);
end
end

function fig = drawMainComparison(base, bad, fOffsetHz, phaseIdeal, cfg)
fig = newFigure(['理想 vs ', bad.label], cfg, [80, 60, 1460, 930]);
annotation(fig, 'textbox', [0.02, 0.955, 0.96, 0.035], ...
    'String', ['理想 IQ vs ', bad.label, '：', bad.note], ...
    'EdgeColor', 'none', 'FontWeight', 'bold', 'FontSize', 14, ...
    'FontName', cfg.fontName, 'Interpreter', 'none');
fMHz = fOffsetHz/1e6;
idealColor = [0.40, 0.45, 0.52];
badColor = [0.85, 0.27, 0.10];
[phiBase, resBase] = phaseDiagnostics( ...
    base.iq, base.iq, phaseIdeal, fOffsetHz, cfg);
[phiBad, resBad] = phaseDiagnostics( ...
    bad.iq, base.iq, phaseIdeal, fOffsetHz, cfg);

subplot(3, 2, 1);
plot(fMHz, real(base.iq), '--', 'Color', [0.30, 0.45, 0.70], 'LineWidth', 1.2);
hold on;
plot(fMHz, imag(base.iq), ':', 'Color', [0.10, 0.58, 0.47], 'LineWidth', 1.4);
plot(fMHz, real(bad.iq), '-', 'Color', [0.85, 0.27, 0.10], 'LineWidth', 1.5);
plot(fMHz, imag(bad.iq), '-.', 'Color', [0.60, 0.28, 0.68], 'LineWidth', 1.5);
title('频域 IQ 分量（统一幅度尺度）');
xlabel('相对中心频率 / MHz'); ylabel('I、Q'); grid on;
legend({'理想 I', '理想 Q', '异常 I', '异常 Q'}, 'Location', 'best');

subplot(3, 2, 2);
plotPair(fMHz, abs(base.iq), abs(bad.iq), idealColor, badColor);
title('频域 IQ 幅度'); xlabel('相对中心频率 / MHz'); ylabel('|IQ|');
ylim([0, max([abs(base.iq); abs(bad.iq)])*1.12]);

subplot(3, 2, 3);
plotPair(fMHz, phiBase, phiBad, idealColor, badColor);
title('展开相位'); xlabel('相对中心频率 / MHz'); ylabel('相位 / rad');

subplot(3, 2, 4);
plotPair(fMHz, resBase, resBad, idealColor, badColor);
title('各自去除最佳线性相位后的残差');
xlabel('相对中心频率 / MHz'); ylabel('残差 / rad');
if max(abs(resBad(isfinite(resBad)))) < 1e-9
    ylim([-0.2, 0.2]);
end

subplot(3, 2, 5);
plotPair(base.cir.distanceM, base.cir.db, bad.cir.db, idealColor, badColor);
title('IFFT CIR：归一化幅度 / dB'); xlabel('双程等效距离 / m');
if ~cfg.roundTrip, xlabel('单程距离 / m'); end
ylabel('20 log10(|CIR| / 自身峰值)');
xlim(cfg.plotDistanceM); ylim([cfg.dbFloor, 1]);
markTruePaths(bad.truePathDistanceM, [cfg.dbFloor, 1]);

subplot(3, 2, 6);
plotPair(base.cir.distanceM, base.cir.normalizedAmplitude, ...
    bad.cir.normalizedAmplitude, idealColor, badColor);
hold on;
near = [min(base.metrics.peakDistanceM, bad.metrics.peakDistanceM)-3.5, ...
    max(base.metrics.peakDistanceM, bad.metrics.peakDistanceM)+3.5];
plot(near, [1, 1]/sqrt(2), ':', 'Color', [0.55, 0.55, 0.55], ...
    'HandleVisibility', 'off');
markWidth(base.metrics, idealColor);
markWidth(bad.metrics, badColor);
markTruePaths(bad.truePathDistanceM, [0, 1.08]);
title(sprintf('功率半高宽：理想 %.3f m；异常 %.3f m', ...
    base.metrics.width3dBM, bad.metrics.width3dBM));
xlabel('距离 / m'); ylabel('各自峰值归一化幅度');
xlim(near); ylim([0, 1.08]);

annotation(fig, 'textbox', [0.02, 0.003, 0.96, 0.032], ...
    'String', sprintf(['合成示例 | %d 点 x %.1f MHz | 两曲线各自峰值归一化 | ', ...
    '半高功率对应幅度 0.707 | 竖虚线标出本例真实路径 | 补零不提高物理分辨率'], ...
    cfg.numTones, cfg.dfHz/1e6), 'EdgeColor', 'none', 'FontSize', 10, ...
    'FontName', cfg.fontName, 'Interpreter', 'none');
end

function fig = drawControls(cases, cfg)
fig = newFigure('补充对照：不是所有 IQ 变化都会使主峰变宽', ...
    cfg, [100, 80, 1530, 880]);
annotation(fig, 'textbox', [0.02, 0.955, 0.96, 0.035], ...
    'String', '补充对照：比较缺点位置、窗函数、线性相位与正常相位折返', ...
    'EdgeColor', 'none', 'FontWeight', 'bold', 'FontSize', 14, ...
    'FontName', cfg.fontName, 'Interpreter', 'none');
groups = {{'edge_missing'}, {'hann_weight', 'hamming_weight'}, ...
    {'center_missing'}, {'linear_phase'}, {'phase_wrap'}};
titles = {'边缘缺点：连续有效频带变窄', 'Hann / Hamming：旁瓣降低、主瓣变宽', ...
    '中间缺点：主瓣未必变宽，旁瓣抬高', '纯线性相位：只改变峰的位置', ...
    '正常 +/-pi 折返：重建 IQ 后 CIR 重合'};
base = cases(1);
colors = [0.85, 0.27, 0.10; 0.10, 0.50, 0.68; 0.10, 0.58, 0.40];
for panel = 1:5
    subplot(2, 3, panel);
    plot(base.cir.distanceM, base.cir.db, '--', ...
        'Color', [0.40, 0.45, 0.52], 'LineWidth', 1.5);
    hold on;
    names = {sprintf('理想：%.3f m', base.metrics.width3dBM)};
    for j = 1:numel(groups{panel})
        index = find(strcmp({cases.id}, groups{panel}{j}), 1);
        item = cases(index);
        plot(item.cir.distanceM, item.cir.db, '-', ...
            'Color', colors(j, :), 'LineWidth', 1.5);
        names{end+1} = sprintf('%s：%.3f m', ...
            item.label, item.metrics.width3dBM); %#ok<AGROW>
    end
    title(titles{panel}); xlabel('距离 / m'); ylabel('归一化 CIR / dB');
    xlim([cfg.distanceM-5, cfg.distanceM+6]); ylim([cfg.dbFloor, 1]);
    grid on; legend(names, 'Location', 'southwest');
end

% 对齐峰位后比较形状，避免把“峰移”误看成“峰宽”。
subplot(2, 3, 6);
plot(base.cir.distanceM-base.metrics.peakDistanceM, ...
    base.cir.normalizedAmplitude, '--', ...
    'Color', [0.40, 0.45, 0.52], 'LineWidth', 1.5);
hold on;
mainIds = {'edge_gaussian', 'quadratic_phase', 'close_multipath'};
names = {'理想单径'};
for j = 1:numel(mainIds)
    index = find(strcmp({cases.id}, mainIds{j}), 1);
    item = cases(index);
    plot(item.cir.distanceM-item.metrics.peakDistanceM, ...
        item.cir.normalizedAmplitude, 'Color', colors(j, :), 'LineWidth', 1.5);
    names{end+1} = item.label; %#ok<AGROW>
end
title('主峰位置对齐后：直接比较三类展宽');
xlabel('相对各自峰位的距离 / m'); ylabel('各自峰值归一化幅度');
xlim([-3.5, 3.5]); ylim([0, 1.08]); grid on;
legend(names, 'Location', 'northeast');
annotation(fig, 'textbox', [0.02, 0.003, 0.96, 0.032], ...
    'String', ['图例数值为功率半高宽；缺失频点在原始网格上置零；', ...
    '每条 CIR 按自身峰值归一化，实际峰高损失见 CSV。'], ...
    'EdgeColor', 'none', 'FontSize', 10, 'FontName', cfg.fontName, ...
    'Interpreter', 'none');
end

function fig = drawAdjacentPhaseDifference(base, bad, ...
    fOffsetHz, phaseIdeal, cfg)
% 专门展示 Delta phase 二次 -> 群时延二次 -> 累积相位约三次 -> CIR。
fig = newFigure('相邻频点相位差二次变化与 CIR 展宽', ...
    cfg, [90, 55, 1460, 930]);
annotation(fig, 'textbox', [0.02, 0.955, 0.96, 0.035], ...
    'String', ['相邻频点残余相位差为开口向下二次函数：', ...
    '对应额外群时延随频率呈二次变化；累积相位误差约为三次函数'], ...
    'EdgeColor', 'none', 'FontWeight', 'bold', 'FontSize', 14, ...
    'FontName', cfg.fontName, 'Interpreter', 'none');
idealColor = [0.40, 0.45, 0.52];
badColor = [0.85, 0.27, 0.10];
fMHz = fOffsetHz/1e6;
[~, residualIdeal] = phaseDiagnostics( ...
    base.iq, base.iq, phaseIdeal, fOffsetHz, cfg);
[~, residualBad] = phaseDiagnostics( ...
    bad.iq, base.iq, phaseIdeal, fOffsetHz, cfg);
[fMidHz, deltaIdeal, delayIdealS] = adjacentPhaseDiagnostics( ...
    base.iq, base.iq, fOffsetHz, cfg);
[~, deltaBad, delayBadS] = adjacentPhaseDiagnostics( ...
    bad.iq, base.iq, fOffsetHz, cfg);

subplot(3, 2, 1);
plotPair(fMidHz/1e6, deltaIdeal, deltaBad, idealColor, badColor);
title('相邻频点残余相位差');
xlabel('相邻频点中点 / MHz'); ylabel('Delta epsilon_k / rad');

subplot(3, 2, 2);
plotPair(fMidHz/1e6, delayIdealS*1e9, delayBadS*1e9, ...
    idealColor, badColor);
title('由相邻相位差得到的额外群时延');
xlabel('相邻频点中点 / MHz'); ylabel('Delta tau_g / ns');

subplot(3, 2, 3);
plotPair(fMHz, residualIdeal, residualBad, idealColor, badColor);
title('去除最佳线性项后的累积相位残差');
xlabel('相对中心频率 / MHz'); ylabel('相位残差 / rad');

subplot(3, 2, 4);
plotPair(fMHz, abs(base.iq), abs(bad.iq), idealColor, badColor);
title('IQ 幅度保持不变');
xlabel('相对中心频率 / MHz'); ylabel('|IQ|'); ylim([0, 1.12]);

subplot(3, 2, 5);
plotPair(base.cir.distanceM, base.cir.db, bad.cir.db, ...
    idealColor, badColor);
title('IFFT CIR：归一化幅度 / dB'); xlabel('距离 / m');
ylabel('20 log10(|CIR| / 自身峰值)');
xlim([cfg.distanceM-7, cfg.distanceM+7]); ylim([cfg.dbFloor, 1]);

subplot(3, 2, 6);
plot(base.cir.distanceM-base.metrics.peakDistanceM, ...
    base.cir.normalizedAmplitude, '--', 'Color', idealColor, ...
    'LineWidth', 1.6); hold on;
plot(bad.cir.distanceM-bad.metrics.peakDistanceM, ...
    bad.cir.normalizedAmplitude, '-', 'Color', badColor, ...
    'LineWidth', 1.7);
plot([-4, 4], [1, 1]/sqrt(2), ':', 'Color', [0.55, 0.55, 0.55], ...
    'HandleVisibility', 'off');
grid on; xlim([-4, 4]); ylim([0, 1.08]);
title(sprintf('峰位对齐后的功率半高宽：%.3f m -> %.3f m', ...
    base.metrics.width3dBM, bad.metrics.width3dBM));
xlabel('相对各自峰位的距离 / m'); ylabel('归一化幅度');
legend({'理想', '相邻相位差二次'}, 'Location', 'best');

annotation(fig, 'textbox', [0.02, 0.003, 0.96, 0.040], ...
    'String', sprintf(['本例 Delta epsilon 峰峰值 %.3f rad，', ...
    '对应额外群时延峰峰值 %.2f ns；宽度比 %.3f。', ...
    '强畸变时可能出现肩峰或分裂，此时单个 -3 dB 宽度不再单调。'], ...
    bad.metrics.adjacentPhaseDiffPpRad, ...
    bad.metrics.extraGroupDelayPpNs, ...
    bad.metrics.width3dBM/base.metrics.width3dBM), ...
    'EdgeColor', 'none', 'FontSize', 10, 'FontName', cfg.fontName, ...
    'Interpreter', 'none');
end

function plotPair(x, yIdeal, yBad, idealColor, badColor)
plot(x, yIdeal, '--', 'Color', idealColor, 'LineWidth', 1.6);
hold on;
plot(x, yBad, '-', 'Color', badColor, 'LineWidth', 1.7);
grid on;
legend({'理想', '本例'}, 'Location', 'best');
end

function markWidth(mt, color)
if isfinite(mt.width3dBM)
    plot([mt.leftCrossingM, mt.rightCrossingM], [1, 1]/sqrt(2), ...
        'o-', 'Color', color, 'LineWidth', 1.8, 'MarkerSize', 4, ...
        'HandleVisibility', 'off');
end
end

function markTruePaths(distances, yRange)
for j = 1:numel(distances)
    plot([distances(j), distances(j)], yRange, ':', ...
        'Color', [0.20, 0.58, 0.45], 'LineWidth', 1.0, ...
        'HandleVisibility', 'off');
end
end

function fig = newFigure(name, cfg, position)
fig = figure('Name', name, 'NumberTitle', 'off', 'Color', 'w', ...
    'Position', position, 'DefaultAxesFontName', cfg.fontName, ...
    'DefaultTextFontName', cfg.fontName, 'DefaultAxesFontSize', 10, ...
    'DefaultTextInterpreter', 'none', 'DefaultLegendInterpreter', 'none');
end

function saveFigure(fig, fileName, cfg)
drawnow;
if cfg.savePng
    set(fig, 'PaperPositionMode', 'auto');
    print(fig, fullfile(cfg.outputDir, [fileName, '.png']), '-dpng', '-r150');
end
if cfg.saveFig
    savefig(fig, fullfile(cfg.outputDir, [fileName, '.fig']));
end
end

function fontName = chooseFont()
fontName = 'Helvetica';
preferred = {'Microsoft YaHei', 'Microsoft YaHei UI', 'SimHei', ...
    'Noto Sans CJK SC', 'PingFang SC'};
available = listfonts;
for j = 1:numel(preferred)
    if any(strcmpi(available, preferred{j}))
        fontName = preferred{j};
        return;
    end
end
end
