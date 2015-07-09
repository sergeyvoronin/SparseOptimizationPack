% plotter for driver driver_Axbvstau_1alg.m run via run_alg_drivers.m 
function plot_driver_Axbvstau_1alg_one_run(data_file, image_dir, settings_file)

% setup image directory
system(['mkdir -p ', image_dir]);

% load data_file with stuff to plot..
load(data_file);

% define tau fracs - may need to adjust num_ticks
tau_fracs_str = {};
num_ticks = 5;
for i=1:num_ticks
    if i==1 
    tau_fracs_str(i) = cellstr(num2str(tau_fracs(i)));
    elseif i==num_ticks
        tau_fracs_str(i) = cellstr(num2str(tau_fracs(end)));
    else
        tau_fracs_str(i) = cellstr('');
    end
end

% load settings
load(settings_file);

    % make plots -->
    alg_name = 'one_alg';
    
    % find intersection point
    intersection_point = 1;
    noise_val = noise_line(1);
    for i=2:length(final_residuals_median)
        %if final_residuals_median(i) < noise_val && final_residuals_median(i-1) >= noise_val
        if (final_residuals_median(i) - noise_val)*(final_residuals_median(i-1) - noise_val) < 0
            intersection_point = i-1;
        end
    end

    % now sum up times and iterations to intersection point
    total_runtimes = sum(total_runtimes_median(1:intersection_point));
    total_iters = sum(num_iters_to_converge_median(1:intersection_point));


    % plot residuals and noise line --->
    final_residuals_median_ht = final_residuals_median(1:intersection_point);
    final_residuals_q1_ht = final_residuals_q1(1:(intersection_point));
    final_residuals_q3_ht = final_residuals_q3(1:(intersection_point));

    fname = [image_dir,'/final_residuals_', alg_name, '.eps'];
    figure(1)
    hold on;
    plot(final_residuals_median,'r--','linewidth',3);
    plot(final_residuals_q1,'g--','linewidth',2);
    plot(final_residuals_q3,'b--','linewidth',2);
    plot(noise_line,'k--','linewidth',3.5);

    plot(final_residuals_median_ht,'r','linewidth',3);
    plot(final_residuals_q1_ht,'g','linewidth',2);
    plot(final_residuals_q3_ht,'b','linewidth',2);

    ylim([0,residuals_y_lim]);
    legend('median','q1','q3','noise');
    xlabel('tau/max(A^t b)');
    ylabel('||Ax - b||_2');
    title('final residuals vs tau');
    
    set(gca,'XTickLabel',tau_fracs_str);
    set(gca,'FontSize',14);
    xl = get(gca,'xlabel'); set(xl,'FontSize',18);
    yl = get(gca,'ylabel'); set(yl,'FontSize',18);
    tit = get(gca,'title'); set(tit,'FontSize',18);
   
    hold off; 
    print('-depsc2',fname);
    close all;


    % plot number of nonzeros and actual sparsity
    num_nnzs_median_ht = num_nnzs_median(1:intersection_point);
    num_nnzs_q1_ht = num_nnzs_q1(1:(intersection_point));
    num_nnzs_q3_ht = num_nnzs_q3(1:(intersection_point));

    fname = [image_dir,'/num_nnzs_', alg_name, '.eps'];
    figure(1000)
    hold on;
    plot(num_nnzs_median,'r--','linewidth',3);
    plot(num_nnzs_q1,'g--','linewidth',2);
    plot(num_nnzs_q3,'b--','linewidth',2);
    plot(num_nnzs_x_median,'m--','linewidth',3.5);

    plot(num_nnzs_median_ht,'r','linewidth',3);
    plot(num_nnzs_q1_ht,'g','linewidth',2);
    plot(num_nnzs_q3_ht,'b','linewidth',2);

    ylim([0,num_nnzs_y_lim]);
    legend('median','q1','q3','input');
    xlabel('tau/max(A^t b)');
    ylabel('number of nonzeros');
    title('number of nonzeros vs tau');
    
    set(gca,'XTickLabel',tau_fracs_str);
    set(gca,'FontSize',14);
    xl = get(gca,'xlabel'); set(xl,'FontSize',18);
    yl = get(gca,'ylabel'); set(yl,'FontSize',18);
    tit = get(gca,'title'); set(tit,'FontSize',18);
   
    hold off; 
    print('-depsc2',fname);
    close all;


    % plot percent errors --->
    percent_errors_median_ht = percent_errors_median(1:intersection_point);
    percent_errors_q1_ht = percent_errors_q1(1:(intersection_point));
    percent_errors_q3_ht = percent_errors_q3(1:(intersection_point));

    fname = [image_dir,'/percent_errors_', alg_name, '.eps'];
    figure(2)
    hold on;
    plot(percent_errors_median,'r--','linewidth',3);
    plot(percent_errors_q1,'g--','linewidth',2);
    plot(percent_errors_q3,'b--','linewidth',2);
    ylim([0,percent_errors_y_lim]);

    plot(percent_errors_median_ht,'r','linewidth',3);
    plot(percent_errors_q1_ht,'g','linewidth',2);
    plot(percent_errors_q3_ht,'b','linewidth',2);

    xlabel('tau/max(A^t b)');
    ylabel('percent error');
    legend('median','q1','q3');
    title('percent errors vs tau');

    set(gca,'XTickLabel',tau_fracs_str);
    set(gca,'FontSize',14);
    xl = get(gca,'xlabel'); set(xl,'FontSize',18);
    yl = get(gca,'ylabel'); set(yl,'FontSize',18);
    tit = get(gca,'title'); set(tit,'FontSize',18);

    hold off;
    print('-depsc2',fname);
    close all;


    % plot number of iterations to converge
    num_iters_to_converge_median_ht = num_iters_to_converge_median(1:intersection_point);

    fname = [image_dir,'/num_iters_', alg_name, '.eps'];
    figure(3)
    hold on;
    plot(num_iters_to_converge_median, 'r--','linewidth',3);
    plot(num_iters_to_converge_median_ht, 'r','linewidth',3);
    %plot(num_iters_to_converge_q1,'g','linewidth',2);
    %plot(num_iters_to_converge_q3,'b','linewidth',2);
    ylim([0,num_iters_y_lim]);
    %legend('median','q1','q3');
    xlabel('tau/max(A^t b)');
    ylabel('number of iteations');
    legend('median');
    title('number of iterations vs tau');

    set(gca,'XTickLabel',tau_fracs_str);
    set(gca,'FontSize',14);
    xl = get(gca,'xlabel'); set(xl,'FontSize',18);
    yl = get(gca,'ylabel'); set(yl,'FontSize',18);
    tit = get(gca,'title'); set(tit,'FontSize',18);

    hold off;
    print('-depsc2',fname);
    close all;


    % plot input and noise match solution
    fname = [image_dir,'/input_and_noise_match_output_', alg_name, '.eps'];
    figure(4)
    hold on;
    plot(x,'r','linewidth',2);
    plot(x_sol_alg_noise_match,'g','linewidth',2);
    legend('input','output');
    title('original input and noise match solution');

    set(gca,'FontSize',14);
    xl = get(gca,'xlabel'); set(xl,'FontSize',18);
    yl = get(gca,'ylabel'); set(yl,'FontSize',18);
    tit = get(gca,'title'); set(tit,'FontSize',18);

    hold off;
    print('-depsc2',fname);
    close all;


    % plot input and best solution
    fname = [image_dir,'/input_and_best_output_', alg_name, '.eps'];
    figure(4)
    hold on;
    plot(x,'r','linewidth',2);
    plot(x_sol_alg_best,'g','linewidth',2);
    legend('input','output');
    title('original input and best reconstruction');

    set(gca,'FontSize',14);
    xl = get(gca,'xlabel'); set(xl,'FontSize',18);
    yl = get(gca,'ylabel'); set(yl,'FontSize',18);
    tit = get(gca,'title'); set(tit,'FontSize',18);

    hold off;
    print('-depsc2',fname);
    close all;


    % plot typical x and svds, these are algorithm independent
    figure(500)
    fname = [image_dir,'/typicalx.eps'];
    h = stem(x,'fill','--');
    set(get(h,'BaseLine'),'LineStyle',':')
    set(h,'MarkerFaceColor','red')
    title('Typical input x');

    set(gca,'FontSize',14);
    xl = get(gca,'xlabel'); set(xl,'FontSize',18);
    yl = get(gca,'ylabel'); set(yl,'FontSize',18);
    tit = get(gca,'title'); set(tit,'FontSize',18);

    print('-depsc2',fname);
    close all;


    figure(600)
    fname = [image_dir,'/typicalsvds.eps'];
    plot(svdsA,'r','linewidth',2);
    title('Typical svds of A');

    set(gca,'FontSize',14);
    xl = get(gca,'xlabel'); set(xl,'FontSize',18);
    yl = get(gca,'ylabel'); set(yl,'FontSize',18);
    tit = get(gca,'title'); set(tit,'FontSize',18);
    
    print('-depsc2',fname);
    close all;

% make bar plots of total runtime and iterations till crossing point (find tau matching noise level)
figure(300)
bar([1],total_runtimes);
ylim([0,total_runtimes_y_lim]);
title('total runtime until match (s)');

set(gca,'FontSize',14);
xl = get(gca,'xlabel'); set(xl,'FontSize',18);
yl = get(gca,'ylabel'); set(yl,'FontSize',18);
tit = get(gca,'title'); set(tit,'FontSize',18);

fname = [image_dir, '/bar_total_runtimes.eps'];
print('-depsc2',fname);
close all;


figure(400)
bar([1],total_iters);
ylim([0,total_iters_y_lim]);
title('total iterations until match');

set(gca,'FontSize',14);
xl = get(gca,'xlabel'); set(xl,'FontSize',18);
yl = get(gca,'ylabel'); set(yl,'FontSize',18);
tit = get(gca,'title'); set(tit,'FontSize',18);

fname = [image_dir, '/bar_total_iters.eps'];
print('-depsc2',fname);
close all;

