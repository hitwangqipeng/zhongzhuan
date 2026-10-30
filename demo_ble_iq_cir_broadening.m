function result = demo_ble_iq_cir_broadening()
%DEMO_BLE_IQ_CIR_BROADENING Compare ideal and impaired frequency-domain IQ.
%
% Usage:
%   result = demo_ble_iq_cir_broadening;
%
% Uses base MATLAB only; Signal Processing and Bluetooth Toolboxes are not
% required. MATLAB R2020a or later is recommended.
%
% Generates six figures:
%   01 edge amplitude attenuation; 02 quadratic phase error;
%   03 unresolved close multipath; 04 control cases;
%   05 quadratic adjacent-tone phase difference;
%   06 two adjacent paths merged into one broadened peak.
% Also saves PNG, FIG, IQ/CIR MAT data, and a CSV of width metrics.
%
% Conventions:
%   1) Each IQ value belongs to one frequency tone in ascending order; the
%      vector is not a time-domain ADC record at one carrier.
%   2) The default is a round-trip synthetic IQ model:
%      phase=-4*pi*f*d/c and distance=c*tau/2.
%   3) The 80 uniformly spaced 1 MHz tones are a synthetic example, not a
%      Bluetooth CS channel map. Endpoint span is 79 MHz; DFT aperture scale
%      is B=N*df=80 MHz.
%   4) All cases share one frequency grid and NFFT. Missing tones are set to
%      zero at their original grid locations.
%   5) No extra window is applied except in the explicit Hann/Hamming cases.
%   6) Each CIR is normalized by its own peak for shape comparison. Absolute
%      peak loss is stored separately in the metrics.
%   7) Width means power FWHM (-3.0103 dB), corresponding to magnitude
%      threshold 1/sqrt(2). Zero padding improves sampling, not resolution.
%   8) Multipath is formed directly as equivalent delayed components in the
%      final IQ. Cross terms from a two-way PCT product H(f)^2 are omitted.

%% User-adjustable parameters
cfg.numTones = 80;
cfg.dfHz = 1e6;
cfg.fcHz = 2.44e9;
cfg.nfft = 16384;
cfg.c = 299792458;
cfg.roundTrip = true;          % true: round-trip IQ; false: one-way IQ
cfg.distanceM = 8;             % ideal path and earliest multipath distance
cfg.commonPhaseRad = 0.4;
cfg.gaussianSigmaFraction = 0.23;  % Gaussian amplitude sigma/B
cfg.phaseQuadraticRad = 2.2;    % approximate quadratic phase at band edges
% This parameter controls the quadratic curvature of the adjacent-tone
% residual phase difference, not the residual phase itself. The model is
% deltaEpsilon(k)=constant-C*x(k)^2, where x is normalized midpoint
% frequency. C>0 gives a downward-opening parabola whose center-to-edge
% difference is about C rad. Removing its discrete mean removes only an
% irrelevant average delay, not the quadratic curvature. Replace the 0.40
% rad demonstration value with the value fitted to measured data.
cfg.adjacentPhaseDiffCurvatureRad = 0.40;
cfg.pathSeparationOverResolution = 0.70;  % path delay spacing divided by 1/B
cfg.mergedPathSeparationOverResolution = 0.90;
cfg.mergedPath2RelativeAmplitude = 1.00;
cfg.mergedPath2RelativePhaseRad = 0.00;
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
    'numTones must be an even integer of at least 8.');
assert(cfg.nfft >= 32*cfg.numTones && mod(cfg.nfft, 2) == 0, ...
    'nfft must be even and at least 32*numTones for stable interpolation.');
assert(cfg.edgeMissingPerSide >= 0 && ...
    2*cfg.edgeMissingPerSide < cfg.numTones, 'Invalid edgeMissingPerSide.');
assert(cfg.centerMissingCount >= 1 && ...
    cfg.centerMissingCount < cfg.numTones, 'Invalid centerMissingCount.');
assert(cfg.mergedPath2RelativeAmplitude >= 0, ...
    'mergedPath2RelativeAmplitude must be nonnegative.');

%% Frequency grid and ideal IQ
N = cfg.numTones;
k = (0:N-1).';
fOffsetHz = (k - (N-1)/2)*cfg.dfHz;
frequencyHz = cfg.fcHz + fOffsetHz;
B = N*cfg.dfHz;
propagationFactor = 1 + double(cfg.roundTrip);
tau0 = propagationFactor*cfg.distanceM/cfg.c;
phaseIdeal = cfg.commonPhaseRad - 2*pi*fOffsetHz*tau0;
iqIdeal = exp(1j*phaseIdeal);

% Use center-relative frequency. The carrier-dependent constant phase is
% absorbed into each complex path gain, allowing paths to be explicitly in
% phase at the center frequency.
cases = struct('id', {}, 'label', {}, 'note', {}, 'iq', {}, ...
    'truePathDistanceM', {}, 'metrics', {}, 'cir', {});
cases(end+1) = makeCase('ideal', 'Ideal single path', ...
    'Flat magnitude plus linear phase', iqIdeal, cfg.distanceM);

%% 1) Amplitude envelope and effective bandwidth
ampGaussian = exp(-0.5*(fOffsetHz/(cfg.gaussianSigmaFraction*B)).^2);
cases(end+1) = makeCase('edge_gaussian', 'Edge amplitude attenuation', ...
    'All tones remain, but lower edge weights reduce effective bandwidth', ...
    iqIdeal.*ampGaussian, cfg.distanceM);

edgeMask = ones(N, 1);
m = cfg.edgeMissingPerSide;
if m > 0
    edgeMask(1:m) = 0;
    edgeMask(N-m+1:N) = 0;
end
cases(end+1) = makeCase('edge_missing', 'Missing edge tones', ...
    'Edge tones are zeroed on the original grid, retaining the center band', ...
    iqIdeal.*edgeMask, cfg.distanceM);

% Implement symmetric windows explicitly to avoid toolbox dependencies.
winHann = 0.5 - 0.5*cos(2*pi*k/(N-1));
winHamming = 0.54 - 0.46*cos(2*pi*k/(N-1));
cases(end+1) = makeCase('hann_weight', 'Frequency-domain Hann weighting', ...
    'Lower sidelobes with a wider main lobe', iqIdeal.*winHann, cfg.distanceM);
cases(end+1) = makeCase('hamming_weight', 'Frequency-domain Hamming weighting', ...
    'Lower sidelobes with a wider main lobe', iqIdeal.*winHamming, cfg.distanceM);

%% 2) Nonlinear phase and linear-phase control
phaseError = cfg.phaseQuadraticRad*(2*fOffsetHz/B).^2;
cases(end+1) = makeCase('quadratic_phase', 'Quadratic phase error', ...
    'Magnitude is unchanged; varying phase slope prevents common-delay focus', ...
    iqIdeal.*exp(1j*phaseError), cfg.distanceM);

% The adjacent-tone residual phase difference is a downward-opening
% quadratic. Since deltaPhaseAdjacent(k)=epsilon(k+1)-epsilon(k), its
% cumulative epsilon(k) is approximately cubic. The discrete group-delay
% increment is
% deltaTau_g(k)=-deltaPhaseAdjacent(k)/(2*pi*df)。
fAdjacentMidHz = 0.5*(fOffsetHz(1:end-1)+fOffsetHz(2:end));
xAdjacent = fAdjacentMidHz/max(abs(fAdjacentMidHz));
deltaPhaseAdjacent = -cfg.adjacentPhaseDiffCurvatureRad*xAdjacent.^2;
deltaPhaseAdjacent = deltaPhaseAdjacent-mean(deltaPhaseAdjacent);
phaseErrorFromAdjacent = [0; cumsum(deltaPhaseAdjacent)];
cases(end+1) = makeCase('quadratic_adjacent_phase_diff', ...
    'Quadratic adjacent-tone phase difference', ...
    ['Downward-opening adjacent-tone residual phase difference; ', ...
    'quadratic group delay and approximately cubic cumulative phase'], ...
    iqIdeal.*exp(1j*phaseErrorFromAdjacent), cfg.distanceM);

deltaTau = propagationFactor*cfg.linearPhaseExtraDistanceM/cfg.c;
cases(end+1) = makeCase('linear_phase', 'Pure linear phase change', ...
    'Control: the peak moves while its width remains nearly unchanged', ...
    iqIdeal.*exp(-1j*2*pi*fOffsetHz*deltaTau), ...
    cfg.distanceM + cfg.linearPhaseExtraDistanceM);

%% 3) Two unresolved nearby paths
pathDeltaTau = cfg.pathSeparationOverResolution/B;
tau1 = tau0;
tau2 = tau0 + pathDeltaTau;
% Both path gains are 0.5 and in phase at center frequency. The earliest
% path remains at distanceM; the combined peak may move and broaden.
iqMultipath = exp(1j*cfg.commonPhaseRad)*( ...
    0.5*exp(-1j*2*pi*fOffsetHz*tau1) + ...
    0.5*exp(-1j*2*pi*fOffsetHz*tau2));
twoDistances = cfg.c*[tau1, tau2]/propagationFactor;
cases(end+1) = makeCase('close_multipath', 'Unresolved nearby multipath', ...
    sprintf('Equal paths in phase at band center; spacing %.2f ns = %.2f/B', ...
    pathDeltaTau*1e9, cfg.pathSeparationOverResolution), ...
    iqMultipath, twoDistances);

% A second, explicit merging example. Each component alone produces the
% normal single-path response. Their separation is below 1/B, so the sum
% has one unresolved maximum with a wider -3 dB main lobe.
mergedDeltaTau = cfg.mergedPathSeparationOverResolution/B;
mergedTau1 = tau0;
mergedTau2 = tau0 + mergedDeltaTau;
mergedGain1 = 1/(1+cfg.mergedPath2RelativeAmplitude);
mergedGain2 = cfg.mergedPath2RelativeAmplitude/ ...
    (1+cfg.mergedPath2RelativeAmplitude)* ...
    exp(1j*cfg.mergedPath2RelativePhaseRad);
iqMergedPath1 = mergedGain1*exp(1j*cfg.commonPhaseRad).* ...
    exp(-1j*2*pi*fOffsetHz*mergedTau1);
iqMergedPath2 = mergedGain2*exp(1j*cfg.commonPhaseRad).* ...
    exp(-1j*2*pi*fOffsetHz*mergedTau2);
iqMergedPaths = iqMergedPath1 + iqMergedPath2;
mergedDistances = cfg.c*[mergedTau1, mergedTau2]/propagationFactor;
cases(end+1) = makeCase('merged_adjacent_paths', ...
    'Two adjacent paths merged into one peak', ...
    sprintf(['Two paths separated by %.2f/B, amplitude ratio %.2f, and ', ...
    'center-frequency phase %.2f rad combine into one broadened peak'], ...
    cfg.mergedPathSeparationOverResolution, ...
    cfg.mergedPath2RelativeAmplitude, cfg.mergedPath2RelativePhaseRad), ...
    iqMergedPaths, mergedDistances);

%% Controls: center-notch tones and normal phase wrapping
centerMask = ones(N, 1);
firstMissing = floor((N-cfg.centerMissingCount)/2) + 1;
centerMask(firstMissing:firstMissing+cfg.centerMissingCount-1) = 0;
cases(end+1) = makeCase('center_missing', 'Missing center tones', ...
    'Control: width need not increase, but sidelobes or false peaks may rise', ...
    iqIdeal.*centerMask, cfg.distanceM);

phaseWrapped = angle(iqIdeal);
iqFromWrappedPhase = abs(iqIdeal).*exp(1j*phaseWrapped);
cases(end+1) = makeCase('phase_wrap', 'Normal wrapped-phase reconstruction', ...
    'Control: wrapping at +/-pi changes neither complex IQ nor the CIR', ...
    iqFromWrappedPhase, cfg.distanceM);

%% Compute all CIRs and metrics
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

%% Mathematical checks for axes, transform sign, and width definition
assert(abs(base.metrics.peakDistanceM-cfg.distanceM) < ...
    2*cfg.c/(propagationFactor*cfg.nfft*cfg.dfHz), ...
    'Ideal-IQ peak does not match distanceM; check IFFT sign and distance axis.');
expectedWidthNs = 0.885893/B*1e9;
assert(abs(base.metrics.width3dBNs/expectedWidthNs-1) < 0.01, ...
    'Ideal rectangular-band power FWHM does not match 0.886/B.');
linearIndex = find(strcmp({cases.id}, 'linear_phase'), 1);
assert(abs(cases(linearIndex).metrics.width3dBM/base.metrics.width3dBM-1) ...
    < 0.005, 'Pure linear phase must not materially change peak width.');
assert(max(abs(iqFromWrappedPhase-iqIdeal)) < 1e-12, ...
    'Reconstruction from normally wrapped phase must preserve complex IQ.');

%% Draw the three primary six-panel comparisons
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
mergedIndex = find(strcmp({cases.id}, 'merged_adjacent_paths'), 1);
fig = drawMergedAdjacentPaths(base, cases(mergedIndex), ...
    iqMergedPath1, iqMergedPath2, cfg);
saveFigure(fig, '06_merged_adjacent_multipath', cfg);

%% Save all frequency IQ, complex CIR, axes, and metrics
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
fprintf('\nSynthetic IQ/CIR comparison complete. Width is power FWHM.\n');
disp(metrics(:, {'label', 'peakDistanceM', 'width3dBNs', ...
    'width3dBM', 'widthRatio', 'peakAmplitudeRatio'}));
fprintf('Output directory: %s\n', cfg.outputDir);
end

function item = makeCase(id, label, note, iq, trueDistances)
item = struct('id', id, 'label', label, 'note', note, 'iq', iq(:), ...
    'truePathDistanceM', trueDistances, 'metrics', [], 'cir', []);
end

function s = computeCir(iq, fOffsetHz, cfg)
N = numel(iq);
% Pass the N IQ values in ascending frequency order directly to IFFT and
% append zero padding. Applying ifftshift before padding would break the
% contiguous-tone ordering.
s.tauS = (-cfg.nfft/2:cfg.nfft/2-1).'/(cfg.nfft*cfg.dfHz);
s.distanceM = cfg.c*s.tauS/(1+double(cfg.roundTrip));
hIndex = fftshift(ifft(iq(:), cfg.nfft))*cfg.nfft/N;
% Restore the phase of the first frequency so the result equals
% sum[IQ_k*exp(j*2*pi*fOffset_k*tau)]/N. This factor does not alter CIR
% magnitude or peak position, but makes the complex CIR match the transform.
s.complex = hIndex.*exp(1j*2*pi*fOffsetHz(1)*s.tauS);
s.amplitude = abs(s.complex);
peak = max(s.amplitude);
assert(peak > 0, 'Input IQ cannot be identically zero.');
s.normalizedAmplitude = s.amplitude/peak;
s.db = 20*log10(max(s.normalizedAmplitude, 10^(cfg.dbFloor/20)));
end

function mt = measurePeak(s, phaseResidual)
y = s.normalizedAmplitude;
[~, p] = max(y);
peakDistance = s.distanceM(p);
% Three-point parabolic interpolation refines only the peak location. Width
% is obtained directly by interpolation at the threshold crossings.
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
% Measure only the contiguous lobe containing the global maximum, so distant
% sidelobes are not included in the width.
if left == 1 || right == numel(y)
    leftTau = NaN;
    rightTau = NaN;
    leftDistance = NaN;
    rightDistance = NaN;
    warning('The main peak crosses the delay-period boundary; width is NaN.');
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
% Phase is undefined for zero-magnitude tones; leave those entries as NaN.
valid = abs(iq) > cfg.phaseValidFraction*max(abs(iq));
phaseUnwrapped = NaN(size(iq));
residual = NaN(size(iq));
if nnz(valid) < 2
    return;
end

% This synthetic test has a known reference. Remove ideal phase before
% unwrapping the residual so gaps cannot cause branch errors on the large
% raw phase slope. Measured data require SNR-aware robust unwrapping.
delta = unwrap(angle(iq(valid).*conj(iqIdeal(valid))));
phaseUnwrapped(valid) = phaseIdeal(valid) + delta;
fMHz = fOffsetHz/1e6;
fitLine = polyfit(fMHz(valid), delta, 1);
residual(valid) = delta-polyval(fitLine, fMHz(valid));
end

function [fMidHz, deltaResidual, extraGroupDelayS] = ...
    adjacentPhaseDiagnostics(iq, iqIdeal, fOffsetHz, cfg)
% Adjacent-tone residual phase difference and inferred extra group delay.
% For uniform tones: tau_g(f_{k+1/2}) ~= -Delta phi_k/(2*pi*Delta f).
ratio = iq(:).*conj(iqIdeal(:));
valid = abs(iq(:)) > cfg.phaseValidFraction*max(abs(iq(:)));
fMidHz = 0.5*(fOffsetHz(1:end-1)+fOffsetHz(2:end));
deltaResidual = NaN(numel(iq)-1, 1);
pairValid = valid(1:end-1) & valid(2:end);
if nnz(valid) < 2
    extraGroupDelayS = deltaResidual;
    return;
end
% Compute the angle of adjacent complex ratios directly, so a magnitude gap
% cannot corrupt later tones through cross-gap unwrapping. The true adjacent
% residual difference must lie in (-pi,pi]. Remove coarse delay first.
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
fig = newFigure(['Ideal vs ', bad.label], cfg, [80, 60, 1460, 930]);
annotation(fig, 'textbox', [0.02, 0.955, 0.96, 0.035], ...
    'String', ['Ideal IQ vs ', bad.label, ': ', bad.note], ...
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
title('Frequency-domain IQ components on a common scale');
xlabel('Frequency offset / MHz'); ylabel('I and Q'); grid on;
legend({'Ideal I', 'Ideal Q', 'Impaired I', 'Impaired Q'}, 'Location', 'best');

subplot(3, 2, 2);
plotPair(fMHz, abs(base.iq), abs(bad.iq), idealColor, badColor);
title('Frequency-domain IQ magnitude'); xlabel('Frequency offset / MHz'); ylabel('|IQ|');
ylim([0, max([abs(base.iq); abs(bad.iq)])*1.12]);

subplot(3, 2, 3);
plotPair(fMHz, phiBase, phiBad, idealColor, badColor);
title('Unwrapped phase'); xlabel('Frequency offset / MHz'); ylabel('Phase / rad');

subplot(3, 2, 4);
plotPair(fMHz, resBase, resBad, idealColor, badColor);
title('Residual after removing each best-fit linear phase');
xlabel('Frequency offset / MHz'); ylabel('Residual / rad');
if max(abs(resBad(isfinite(resBad)))) < 1e-9
    ylim([-0.2, 0.2]);
end

subplot(3, 2, 5);
plotPair(base.cir.distanceM, base.cir.db, bad.cir.db, idealColor, badColor);
title('IFFT CIR: normalized magnitude / dB'); xlabel('Round-trip equivalent distance / m');
if ~cfg.roundTrip, xlabel('One-way distance / m'); end
ylabel('20 log10(|CIR| / own peak)');
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
title(sprintf('Power FWHM: ideal %.3f m; impaired %.3f m', ...
    base.metrics.width3dBM, bad.metrics.width3dBM));
xlabel('Distance / m'); ylabel('Magnitude normalized by own peak');
xlim(near); ylim([0, 1.08]);

annotation(fig, 'textbox', [0.02, 0.003, 0.96, 0.032], ...
    'String', sprintf(['Synthetic example | %d tones x %.1f MHz | Each curve ', ...
    'normalized by its own peak | Half power = 0.707 magnitude | ', ...
    'Vertical dotted lines are true paths | Zero padding does not improve resolution'], ...
    cfg.numTones, cfg.dfHz/1e6), 'EdgeColor', 'none', 'FontSize', 10, ...
    'FontName', cfg.fontName, 'Interpreter', 'none');
end

function fig = drawControls(cases, cfg)
fig = newFigure('Controls: not every IQ change broadens the main peak', ...
    cfg, [100, 80, 1530, 880]);
annotation(fig, 'textbox', [0.02, 0.955, 0.96, 0.035], ...
    'String', 'Controls: tone location, windows, linear phase, and normal phase wrapping', ...
    'EdgeColor', 'none', 'FontWeight', 'bold', 'FontSize', 14, ...
    'FontName', cfg.fontName, 'Interpreter', 'none');
groups = {{'edge_missing'}, {'hann_weight', 'hamming_weight'}, ...
    {'center_missing'}, {'linear_phase'}, {'phase_wrap'}};
titles = {'Missing edge tones: narrower contiguous effective band', ...
    'Hann / Hamming: lower sidelobes and wider main lobe', ...
    'Missing center tones: width need not increase; sidelobes rise', ...
    'Pure linear phase: only the peak position changes', ...
    'Normal +/-pi wrapping: reconstructed IQ gives the same CIR'};
base = cases(1);
colors = [0.85, 0.27, 0.10; 0.10, 0.50, 0.68; 0.10, 0.58, 0.40];
for panel = 1:5
    subplot(2, 3, panel);
    plot(base.cir.distanceM, base.cir.db, '--', ...
        'Color', [0.40, 0.45, 0.52], 'LineWidth', 1.5);
    hold on;
    names = {sprintf('Ideal: %.3f m', base.metrics.width3dBM)};
    for j = 1:numel(groups{panel})
        index = find(strcmp({cases.id}, groups{panel}{j}), 1);
        item = cases(index);
        plot(item.cir.distanceM, item.cir.db, '-', ...
            'Color', colors(j, :), 'LineWidth', 1.5);
        names{end+1} = sprintf('%s：%.3f m', ...
            item.label, item.metrics.width3dBM); %#ok<AGROW>
    end
    title(titles{panel}); xlabel('Distance / m'); ylabel('Normalized CIR / dB');
    xlim([cfg.distanceM-5, cfg.distanceM+6]); ylim([cfg.dbFloor, 1]);
    grid on; legend(names, 'Location', 'southwest');
end

% Align peak positions before shape comparison so motion is not mistaken for
% broadening.
subplot(2, 3, 6);
plot(base.cir.distanceM-base.metrics.peakDistanceM, ...
    base.cir.normalizedAmplitude, '--', ...
    'Color', [0.40, 0.45, 0.52], 'LineWidth', 1.5);
hold on;
mainIds = {'edge_gaussian', 'quadratic_phase', 'close_multipath'};
names = {'Ideal single path'};
for j = 1:numel(mainIds)
    index = find(strcmp({cases.id}, mainIds{j}), 1);
    item = cases(index);
    plot(item.cir.distanceM-item.metrics.peakDistanceM, ...
        item.cir.normalizedAmplitude, 'Color', colors(j, :), 'LineWidth', 1.5);
    names{end+1} = item.label; %#ok<AGROW>
end
title('Peak-aligned comparison of three broadening mechanisms');
xlabel('Distance relative to each peak / m'); ylabel('Normalized magnitude');
xlim([-3.5, 3.5]); ylim([0, 1.08]); grid on;
legend(names, 'Location', 'northeast');
annotation(fig, 'textbox', [0.02, 0.003, 0.96, 0.032], ...
    'String', ['Legend values are power FWHM. Missing tones are zeroed on ', ...
    'the original grid. Each CIR uses its own peak; absolute peak loss is in the CSV.'], ...
    'EdgeColor', 'none', 'FontSize', 10, 'FontName', cfg.fontName, ...
    'Interpreter', 'none');
end

function fig = drawAdjacentPhaseDifference(base, bad, ...
    fOffsetHz, phaseIdeal, cfg)
% Show quadratic Delta phase -> quadratic group delay -> approximately cubic
% cumulative phase -> CIR.
fig = newFigure('Quadratic adjacent-tone phase difference and CIR broadening', ...
    cfg, [90, 55, 1460, 930]);
annotation(fig, 'textbox', [0.02, 0.955, 0.96, 0.035], ...
    'String', ['Downward-opening quadratic adjacent-tone residual phase ', ...
    'difference: quadratic extra group delay and approximately cubic cumulative phase'], ...
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
title('Adjacent-tone residual phase difference');
xlabel('Adjacent-tone midpoint / MHz'); ylabel('Delta epsilon_k / rad');

subplot(3, 2, 2);
plotPair(fMidHz/1e6, delayIdealS*1e9, delayBadS*1e9, ...
    idealColor, badColor);
title('Extra group delay inferred from adjacent phase');
xlabel('Adjacent-tone midpoint / MHz'); ylabel('Delta tau_g / ns');

subplot(3, 2, 3);
plotPair(fMHz, residualIdeal, residualBad, idealColor, badColor);
title('Cumulative phase residual after best linear fit');
xlabel('Frequency offset / MHz'); ylabel('Phase residual / rad');

subplot(3, 2, 4);
plotPair(fMHz, abs(base.iq), abs(bad.iq), idealColor, badColor);
title('IQ magnitude remains unchanged');
xlabel('Frequency offset / MHz'); ylabel('|IQ|'); ylim([0, 1.12]);

subplot(3, 2, 5);
plotPair(base.cir.distanceM, base.cir.db, bad.cir.db, ...
    idealColor, badColor);
title('IFFT CIR: normalized magnitude / dB'); xlabel('Distance / m');
ylabel('20 log10(|CIR| / own peak)');
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
title(sprintf('Peak-aligned power FWHM: %.3f m -> %.3f m', ...
    base.metrics.width3dBM, bad.metrics.width3dBM));
xlabel('Distance relative to each peak / m'); ylabel('Normalized magnitude');
legend({'Ideal', 'Quadratic adjacent-phase difference'}, 'Location', 'best');

annotation(fig, 'textbox', [0.02, 0.003, 0.96, 0.040], ...
    'String', sprintf(['Delta epsilon peak-to-peak %.3f rad; extra group-delay ', ...
    'peak-to-peak %.2f ns; width ratio %.3f. Strong distortion may create ', ...
    'shoulders or split peaks, making a single -3 dB width non-monotonic.'], ...
    bad.metrics.adjacentPhaseDiffPpRad, ...
    bad.metrics.extraGroupDelayPpNs, ...
    bad.metrics.width3dBM/base.metrics.width3dBM), ...
    'EdgeColor', 'none', 'FontSize', 10, 'FontName', cfg.fontName, ...
    'Interpreter', 'none');
end

function fig = drawMergedAdjacentPaths(base, merged, iqPath1, iqPath2, cfg)
% Show why two nearby path responses are observed as one wider peak.
cir1 = computeCir(iqPath1, ...
    ((0:cfg.numTones-1).'-(cfg.numTones-1)/2)*cfg.dfHz, cfg);
cir2 = computeCir(iqPath2, ...
    ((0:cfg.numTones-1).'-(cfg.numTones-1)/2)*cfg.dfHz, cfg);
fig = newFigure('Two adjacent paths merge into a broadened CIR peak', ...
    cfg, [95, 60, 1460, 900]);
annotation(fig, 'textbox', [0.02, 0.955, 0.96, 0.035], ...
    'String', sprintf(['Two paths are separated by %.2f/B with amplitude ', ...
    'ratio %.2f and center-frequency phase %.2f rad. They form one wider peak.'], ...
    cfg.mergedPathSeparationOverResolution, ...
    cfg.mergedPath2RelativeAmplitude, cfg.mergedPath2RelativePhaseRad), ...
    'EdgeColor', 'none', 'FontWeight', 'bold', 'FontSize', 14, ...
    'FontName', cfg.fontName, 'Interpreter', 'none');

subplot(2, 2, 1);
plot(cir1.distanceM, cir1.normalizedAmplitude, '--', ...
    'Color', [0.10, 0.50, 0.75], 'LineWidth', 1.5); hold on;
plot(cir2.distanceM, cir2.normalizedAmplitude, '--', ...
    'Color', [0.20, 0.65, 0.40], 'LineWidth', 1.5);
plot(merged.cir.distanceM, merged.cir.normalizedAmplitude, '-', ...
    'Color', [0.85, 0.27, 0.10], 'LineWidth', 2.0);
markTruePaths(merged.truePathDistanceM, [0, 1.08]);
xlim([cfg.distanceM-3, max(merged.truePathDistanceM)+3]); ylim([0, 1.08]);
grid on; title('Individual path responses and their coherent sum');
xlabel('Distance / m'); ylabel('Normalized magnitude');
legend({'Path 1 alone', 'Path 2 alone', 'Coherent two-path sum'}, ...
    'Location', 'best');

subplot(2, 2, 2);
plot(base.cir.distanceM-base.metrics.peakDistanceM, ...
    base.cir.normalizedAmplitude, '--', 'Color', [0.40, 0.45, 0.52], ...
    'LineWidth', 1.6); hold on;
plot(merged.cir.distanceM-merged.metrics.peakDistanceM, ...
    merged.cir.normalizedAmplitude, '-', 'Color', [0.85, 0.27, 0.10], ...
    'LineWidth', 1.9);
plot([-4, 4], [1, 1]/sqrt(2), ':', 'Color', [0.55, 0.55, 0.55], ...
    'HandleVisibility', 'off');
shiftedBaseMetrics = base.metrics;
shiftedBaseMetrics.leftCrossingM = base.metrics.leftCrossingM- ...
    base.metrics.peakDistanceM;
shiftedBaseMetrics.rightCrossingM = base.metrics.rightCrossingM- ...
    base.metrics.peakDistanceM;
markWidth(shiftedBaseMetrics, [0.40, 0.45, 0.52]);
shiftedMetrics = merged.metrics;
shiftedMetrics.leftCrossingM = merged.metrics.leftCrossingM- ...
    merged.metrics.peakDistanceM;
shiftedMetrics.rightCrossingM = merged.metrics.rightCrossingM- ...
    merged.metrics.peakDistanceM;
markWidth(shiftedMetrics, [0.85, 0.27, 0.10]);
xlim([-4, 4]); ylim([0, 1.08]); grid on;
title(sprintf('Peak-aligned -3 dB width: %.3f m -> %.3f m', ...
    base.metrics.width3dBM, merged.metrics.width3dBM));
xlabel('Distance relative to each peak / m'); ylabel('Normalized magnitude');
legend({'Ideal single path', 'Merged adjacent paths'}, 'Location', 'best');

subplot(2, 2, 3);
plot(merged.cir.distanceM, merged.cir.db, '-', ...
    'Color', [0.85, 0.27, 0.10], 'LineWidth', 1.7); hold on;
markTruePaths(merged.truePathDistanceM, [cfg.dbFloor, 1]);
xlim([cfg.distanceM-5, max(merged.truePathDistanceM)+5]);
ylim([cfg.dbFloor, 1]); grid on;
title('Merged CIR on a logarithmic scale');
xlabel('Distance / m'); ylabel('Normalized CIR / dB');

subplot(2, 2, 4);
fMHz = ((0:cfg.numTones-1).'-(cfg.numTones-1)/2)*cfg.dfHz/1e6;
plot(fMHz, abs(iqPath1), '--', ...
    'Color', [0.10, 0.50, 0.75], 'LineWidth', 1.4); hold on;
plot(fMHz, abs(iqPath2), '--', ...
    'Color', [0.20, 0.65, 0.40], 'LineWidth', 1.4);
plot(fMHz, abs(merged.iq), '-', ...
    'Color', [0.85, 0.27, 0.10], 'LineWidth', 1.8);
grid on; title('Frequency-domain magnitude interference');
xlabel('Frequency offset / MHz'); ylabel('|IQ|');
legend({'Path 1 component', 'Path 2 component', 'Two-path sum'}, ...
    'Location', 'best');

annotation(fig, 'textbox', [0.02, 0.003, 0.96, 0.038], ...
    'String', sprintf(['Path spacing %.3f m; ideal width %.3f m; ', ...
    'merged width %.3f m; width ratio %.3f.'], ...
    diff(merged.truePathDistanceM), base.metrics.width3dBM, ...
    merged.metrics.width3dBM, merged.metrics.width3dBM/base.metrics.width3dBM), ...
    'EdgeColor', 'none', 'FontSize', 10, 'FontName', cfg.fontName, ...
    'Interpreter', 'none');
end

function plotPair(x, yIdeal, yBad, idealColor, badColor)
plot(x, yIdeal, '--', 'Color', idealColor, 'LineWidth', 1.6);
hold on;
plot(x, yBad, '-', 'Color', badColor, 'LineWidth', 1.7);
grid on;
legend({'Ideal', 'Current case'}, 'Location', 'best');
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
