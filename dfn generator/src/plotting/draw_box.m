function draw_box(modelBox)

    x = double(modelBox.dx);
    y = double(modelBox.dy);
    z = double(modelBox.dz);

    plot3([0 x x 0 0],[0 0 y y 0],[0 0 0 0 0],'k-','LineWidth',1);
    plot3([0 x x 0 0],[0 0 y y 0],[z z z z z],'k-','LineWidth',1);

    for xx = [0 x]
        for yy = [0 y]
            plot3([xx xx],[yy yy],[0 z],'k-','LineWidth',1);
        end
    end
end