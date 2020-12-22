function labelTable = findPQRST(x,t,parentLabelVal,parentLabelLoc,varargin)
% This is a template for creating a custom function for automated labeling
%
%  x is a matrix where each column contains data corresponding to a
%  channel. If the channels have different lengths, then x is a cell array
%  of column vectors.
%
%  t is a matrix where each column contains time corresponding to a
%  channel. If the channels have different lengths, then t is a cell array
%  of column vectors.
%
%  parentLabelVal is the parent label value associated with the output
%  sublabel or empty when output is not a sublabel.
%  parentLabelLoc contains an empty vector when the parent label is an
%  attribute, a vector of ROI limits when parent label is an ROI or a point
%  location when parent label is a point.
%
%  labelVals must be a column vector with numeric, logical or string output
%  values.
%  labelLocs must be an empty vector when output labels are attributes, a
%  two column matrix of ROI limits when output labels are ROIs, or a column
%  vector of point locations when output labels are points.

labelVals = cell(2,1);
labelLocs = cell(2,1);

if nargin<5
    Fs = 250;
else
    Fs = varargin{1};
end

df = 15;

load('trainedQTSegmentationNetwork','net')

for kj = 1:size(x,2)

    sig = x(:,kj);
      
    % Reshape input and compute Fourier synchrosqueezed transforms

    mitFSST = computeFSST(sig,Fs);
    
    % Use trained network to predict which points belong to QRS regions
    
    netPreds = classify(net,mitFSST,'MiniBatchSize',50);

    % Create a signal mask for QRS regions and specify minimum sequence length
    
    QRS = categorical([netPreds{1} netPreds{2}]);
    msk = signalMask(QRS,"MinLength",df,"SampleRate",Fs);
    r = roimask(msk);
    
    % Label QRS complexes as regions of interest
    
    labelVals{kj} = r.Value;
    labelLocs{kj} = r.ROILimits;

end

labelVals = vertcat(labelVals{:});
labelLocs = cell2mat(labelLocs);
labelTable = table(labelLocs,labelVals);

function signalsFsst = computeFSST(xd,Fs)

xd = reshape([xd;randn(10000-length(xd),1)/100],5000,2);
signalsFsst = cell(1,2);    
    
for k = 1:2
    [ss,ff] = fsst(xd(:,k),Fs,kaiser(128));
    sp = ss(ff>0.5 & ff<40,:);
    signalsFsst{k} = normalize([real(sp);imag(sp)],2);
end

end

function [locs,vals] = p2qrs(k)

fc = 1e-6;
df = 20;

ctgs = categories(k.Value);
levs = 1:length(ctgs);
for jk = levs
   cat2num(k.Value == ctgs{jk}) = levs(jk);
end
chpt = findchangepts(cat2num,'MaxNumChanges',length(cat2num));
locs = [[1;chpt'] [chpt'-1;length(cat2num)]+fc];

vals = categorical(cat2num(locs(:,1))',levs,ctgs);
locs = locs+round(k.Location(1))-1;

qrs = find(vals=='QRS' & diff(locs,[],2)>df);

vals = categorical(string(vals(qrs)),["QRS" "n/a"]);

locs = locs(qrs,:);

end

end


