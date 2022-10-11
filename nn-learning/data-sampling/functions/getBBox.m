function bbox = getBBox(V)
    bbox.xmin = min(V(:,1));
    bbox.xmax = max(V(:,1));
    bbox.ymin = min(V(:,2));
    bbox.ymax = max(V(:,2));
    bbox.zmin = min(V(:,3));
    bbox.zmax = max(V(:,3));
end

