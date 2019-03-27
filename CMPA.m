close all
clear

%Set up diode parameters
Is = 0.01E-12; %A
Ib = 0.1E-12; %A
Vb = 1.3; %V
Gp = 0.1; %ohm^-1

%Define noise variation percentage, apply half to each side, just change
%noiseVar to change variation width
noiseVar = 0.4;
noiseLower = 1-(noiseVar./2);
noiseUpper = 1+(noiseVar./2);

diodeI = @(V, Is, Ib, Vb, Gp) Is.*(exp((1.2./0.025).*V)-1) + Gp.*V + Ib.*(exp((-1.2./0.25).*(V + Vb)));

%Get current vector
deviceV = linspace(-1.95, 0.7, 200); %V
I = diodeI(deviceV, Is, Ib, Vb, Gp);

%Get noisy current vector by getting variation percent
INoise = rand(1,length(deviceV));
INoise = (noiseUpper-noiseLower).*INoise + noiseLower;
INoise = INoise.*I;

order4 = polyfit(deviceV, I, 4);
order4 = polyval(order4, deviceV);
order8 = polyfit(deviceV, I, 8);
order8 = polyval(order8, deviceV);

figure()
subplot(2,1,1);
plot(deviceV, I)
hold on
plot(deviceV, order4, 'r-')
plot(deviceV, order8, 'g-')
title('No Variation Data Using Plot');
subplot(2,1,2);
semilogy(deviceV, abs(I))
hold on
semilogy(deviceV, abs(order4))
semilogy(deviceV, abs(order8))

title('No Variation Data using semilogy');
legend('Generated', 'Order 4', 'Order 8')

order4Noise = polyfit(deviceV, I, 4);
order4Noise = polyval(order4Noise, deviceV);
order8Noise = polyfit(deviceV, I, 8);
order8Noise = polyval(order8Noise, deviceV);


figure()
subplot(2,1,1);
plot(deviceV, INoise)
hold on
plot(deviceV, order4Noise, 'r-')
plot(deviceV, order8Noise, 'g-')
title('Variation Data Using Plot');
subplot(2,1,2);
semilogy(deviceV, abs(INoise))
hold on
semilogy(deviceV, abs(order4Noise), 'r-')
semilogy(deviceV, abs(order8Noise), 'g-')
title('Variation Data using semilogy');
legend('Generated', 'Order 4', 'Order 8')


fo1 = fittype('A.*(exp(1.2*x/25e-3)-1) + 0.1.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff1=fit(deviceV',I',fo1);
If1=ff1(deviceV);


fo2 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff2=fit(deviceV',I',fo2);
If2=ff2(deviceV);



fo3 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');
ff3=fit(deviceV',I',fo3);
If3=ff3(deviceV);



figure()
plot(deviceV,If1)
hold on
plot(deviceV,If2)
plot(deviceV,If3)
legend('fit 1','fit 2','fit 3')
title('Nonlinear curve fitting')

figure()
semilogy(deviceV,abs(If1))
hold on
semilogy(deviceV,abs(If2))
semilogy(deviceV,abs(If3))
legend('fit 1','fit 2','fit 3')
title('Nonlinear curve fitting (semilog)')


inputs = deviceV.'; 
targets = I.'; 
hiddenLayerSize = 10; 
net = fitnet(hiddenLayerSize); 
net.divideParam.trainRatio = 70/100; 
net.divideParam.valRatio = 15/100; 
net.divideParam.testRatio = 15/100; 
[net,tr] = train(net,inputs,targets); 
outputs = net(inputs); 
errors = gsubtract(outputs,targets); 
performance = perform(net,targets,outputs) 
view(net) 
Inn = outputs;

figure()
plot(deviceV, I, 'x');
hold on
plot(deviceV, Inn);
title('Neural Net Fit')
legend('Generated', 'Neural Net');

figure()
semilogy(deviceV, abs(I), 'x');
hold on 
semilogy(deviceV, abs(Inn));
title('Neural Net Fit')
legend('Generated', 'Neural Net');


