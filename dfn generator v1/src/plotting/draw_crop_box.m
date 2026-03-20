function draw_crop_box(cropBox)

    cx = (cropBox.xmin + cropBox.xmax)/2;
    cy = (cropBox.ymin + cropBox.ymax)/2;
    cz = (cropBox.zmin + cropBox.zmax)/2;

    x1 = cropBox.xmin - cx;
    x2 = cropBox.xmax - cx;

    y1 = cropBox.ymin - cy;
    y2 = cropBox.ymax - cy;

    z1 = cropBox.zmin - cz;
    z2 = cropBox.zmax - cz;

    plot3([x1 x2 x2 x1 x1],[y1 y1 y2 y2 y1],[z1 z1 z1 z1 z1],'k-');
    plot3([x1 x2 x2 x1 x1],[y1 y1 y2 y2 y1],[z2 z2 z2 z2 z2],'k-');

    for xx = [x1 x2]
        for yy = [y1 y2]
            plot3([xx xx],[yy yy],[z1 z2],'k-');
        end
    end
end