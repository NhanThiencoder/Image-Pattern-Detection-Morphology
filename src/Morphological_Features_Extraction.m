%% MORPHOLOGICAL FEATURE EXTRACTION FOR LEAF IMAGES (MASK-BASED)
warning('off', 'all');
script_dir = fileparts(mfilename('fullpath'));

% Kiem tra Image Processing Toolbox truoc khi chay
if ~license('test', 'image_toolbox') || isempty(ver('images'))
    error(['Thieu Image Processing Toolbox. Vui long cai dat/enable de su dung ', ...
        'imfill, bwareaopen, bwconncomp, regionprops.']);
end

segmented_dir = fullfile(script_dir, '..', 'data', 'segmented', 'dataset_segmented');
processed_dir = fullfile(script_dir, '..', 'data', 'processed');

if ~exist(segmented_dir, 'dir')
    error(['Khong tim thay du lieu tai: ', segmented_dir, ...
           newline, 'Hay kiem tra lai xem thu muc data co nam cung cap voi models khong.']);
end

if ~exist(processed_dir, 'dir')
    mkdir(processed_dir);
end

imds_seg = imageDatastore(segmented_dir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

num_images = numel(imds_seg.Files);
features_table = table();

h = waitbar(0, 'Dang trich xuat Morphological Features... Vui long doi');

for i = 1:num_images
    img_path = imds_seg.Files{i};
    label = string(imds_seg.Labels(i));

    img = imread(img_path);
    img = imresize(img, [256 256]);

    % Mask tu nen den (0,0,0)
    mask = (img(:,:,1) > 0) | (img(:,:,2) > 0) | (img(:,:,3) > 0);
    if sum(mask(:)) == 0
        continue;
    end

    % Lam sach mask
    mask = imfill(mask, 'holes');
    mask = bwareaopen(mask, 50);

    % Chi giu vung lon nhat neu co nhieu vung
    cc = bwconncomp(mask);
    if cc.NumObjects > 1
        stats_area = regionprops(cc, 'Area');
        [~, idx] = max([stats_area.Area]);
        mask = false(size(mask));
        mask(cc.PixelIdxList{idx}) = true;
    end

    % Tinh toan dac trung hinh thai
    props = regionprops(mask, 'Area', 'Perimeter', 'Eccentricity', ...
        'Solidity', 'Extent', 'MajorAxisLength', 'MinorAxisLength', ...
        'EquivDiameter', 'ConvexArea');

    if isempty(props)
        continue;
    end

    area = props.Area;
    perimeter = props.Perimeter;
    eccentricity = props.Eccentricity;
    solidity = props.Solidity;
    extent = props.Extent;
    major_axis = props.MajorAxisLength;
    minor_axis = props.MinorAxisLength;
    equiv_diameter = props.EquivDiameter;
    convex_area = props.ConvexArea;

    aspect_ratio = major_axis / max(minor_axis, eps);
    circularity = (4 * pi * area) / max(perimeter^2, eps);

    new_row = table(label, area, perimeter, eccentricity, solidity, extent, ...
        major_axis, minor_axis, aspect_ratio, equiv_diameter, convex_area, circularity, ...
        'VariableNames', {'Label', ...
        'Area', 'Perimeter', 'Eccentricity', 'Solidity', 'Extent', ...
        'MajorAxisLength', 'MinorAxisLength', 'AspectRatio', 'EquivDiameter', 'ConvexArea', 'Circularity'});

    features_table = [features_table; new_row];

    if mod(i, 100) == 0
        waitbar(i / num_images, h, sprintf('Dang xu ly: %d / %d anh...', i, num_images));
    end
end

close(h);

output_csv = fullfile(processed_dir, 'dataset_morphology.csv');
writetable(features_table, output_csv);

disp('HOAN TAT!');
disp(['Da luu ', num2str(height(features_table)), ' dong du lieu vao file: ', output_csv]);
