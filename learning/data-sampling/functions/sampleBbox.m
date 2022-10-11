function points = sampleBbox(bbox, n_pts)
    points = rand(n_pts,3);
    points(:,1) = bbox.xmin+(bbox.xmax - bbox.xmin)*points(:,1);
    points(:,2) = bbox.ymin+(bbox.ymax - bbox.ymin)*points(:,2);
    points(:,3) = bbox.zmin+(bbox.zmax - bbox.zmin)*points(:,3);
end