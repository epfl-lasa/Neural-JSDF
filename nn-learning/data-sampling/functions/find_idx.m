function idx = find_idx(body_name,mesh)
idx = 0;
for i = 1:1:length(mesh)
    if strcmp(mesh{i}.name,body_name)
        idx = i;
        break;
    end
end

