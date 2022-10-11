function bbox_scaled = scaleBBox(bbox, d)
    bbox_scaled.xmin = bbox.xmin - d;
    bbox_scaled.xmax = bbox.xmax + d;
    bbox_scaled.ymin = bbox.ymin - d;
    bbox_scaled.ymax = bbox.ymax + d;
    bbox_scaled.zmin = bbox.zmin - d;
    bbox_scaled.zmax = bbox.zmax + d;
end

