function [main,net,ANN] =ANNCreator()
%% Load input and output data and creat TF of the system %%%%%%%%%%%%%%%%%%
[n,d]=pade(0.04,2);
sys=tf(70,[0.17 1])*tf(n,d);
clear n d

load('MultiC_Step_Neg')
u1=Inp.signals.values';
y1=Out.signals.values';
t1=Inp.time';

load('MultiC_Step_Pos')
u2=Inp.signals.values';
y2=Out.signals.values';
t2=max(t1)+0.001+Inp.time';

load('MultiD_Step_Neg')
u3=Inp.signals.values';
y3=Out.signals.values';
t3=max(t2)+0.001+Inp.time';

load('MultiD_Step_Pos')
u4=Inp.signals.values';
y4=Out.signals.values';
t4=max(t3)+0.001+Inp.time';

load('MultiE_Step_Neg')
u5=Inp.signals.values';
y5=Out.signals.values';
t5=max(t4)+0.001+Inp.time';

load('MultiE_Step_Pos')
u6=Inp.signals.values';
y6=Out.signals.values';
t6=max(t5)+0.001+Inp.time';

main.Inp=[u1 u2 u3 u4];
main.Tar=[y1 y2 y3 y4];
main.Time=[t1 t2 t3 t4];

[par,ty]=lsim(sys,main.Inp,main.Time);
main.TF=par';

clear data ty par test_data u1 u2 u3 u4 y1 y2 y3 y4 y5 y6 t1 t2 t3 t4 t5 t6

%% Define ANN properties %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Layer structure, delays, and epochs
prompt = {'Layer structure, Str= ','Input delay, DelInp=','Feedback delay, DelFe=','Epoch limit, Epo='};

dlg_title = 'Input data';
num_lines = 1;
def = {'[5]','0:1:5','1:1:5','100'};

answer=inputdlg(prompt,dlg_title,num_lines,def);

ANN.Str =str2num(answer{1,:});
ANN.DelInp =str2num(answer{2,:});
ANN.DelFe =str2num(answer{3,:});
ANN.Epo =str2num(answer{4,:});

clear prompt dlg_title num_lines def answer

%% Select ANN input %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
choice.a = questdlg('Please choose the ANN input:','Select ANN input','{IN;TF}','[IN;TF]','{IN}','[IN]');

switch choice.a
    case '{IN}'
        ANN.Inp=con2seq(main.Inp);
        ANN.Tar=con2seq(main.Tar);

    case '[IN;TF]'
        ANN.Inp=[main.Inp;main.TF];
        ANN.Tar=main.Tar;

    case '{IN;TF}'
        ANN.Inp=con2seq([main.Inp;main.TF]);
        ANN.Tar=con2seq(main.Tar);

    otherwise '[IN]'
        ANN.Inp=main.Inp;
        ANN.Tar=main.Tar;
end

%% Select ANN type: newff, newelm, or newnarx %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
choice.c = questdlg('Please choose an ANN type:', ...
	'Select ANN type', ...
	'newff','newelm','newnarx','newnarx');

switch choice.c

    case 'newff'
        net=newff(ANN.Inp,ANN.Tar,ANN.Str);	
        net.inputWeights{1,1}.delays=ANN.DelInp;

    case 'newelm'
        net=newelm(ANN.Inp,ANN.Tar,ANN.Str);
        net.inputWeights{1,1}.delays=ANN.DelInp;
        net.layerWeights{1,length(ANN.Str)}.delays=ANN.DelFe; 

    case 'newnarx'
        net=newnarx(ANN.Inp,ANN.Tar,ANN.DelInp,ANN.DelFe,ANN.Str);
        net.inputWeights{1,1}.delays=ANN.DelInp;
        net.layerWeights{1,length(ANN.Str)}.delays=ANN.DelFe;
end

clear choice.c

%% Train the network
net.trainParam.mu_max=1.0e20; % Important parameter
net.trainParam.epochs = ANN.Epo;

[net,main.tr] = train(net,ANN.Inp,ANN.Tar);

%% Test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('MultiE_Step_Pos')

main.test_Inp=Inp.signals.values';
main.test_Tar=Out.signals.values';
main.test_Time=Inp.time';

[par,ty]=lsim(sys,main.test_Inp,main.test_Time);
main.test_TF=par';

clear ty par

switch choice.a

    case '[IN]'
        ANN.test_Inp=main.test_Inp;
        ANN.test_Tar=main.test_Tar;

    case '[IN;TF]'
        ANN.test_Inp=[main.test_Inp;main.test_TF];
        ANN.test_Tar=main.test_Tar;

    case '{IN;TF}'
        ANN.test_Inp=con2seq([main.test_Inp;main.test_TF]);
        ANN.test_Tar=con2seq(main.test_Tar);

    otherwise  %{IN}
        ANN.test_Inp=con2seq(main.test_Inp);
        ANN.test_Tar=con2seq(main.test_Tar);
end

Y = sim(net,ANN.test_Inp);

if iscell(Y)
    Y=cell2mat(Y);
end
    
figure
plot(main.test_Time,main.test_Tar,'g',main.test_Time,Y,'r'),
legend("Real System Response", "ANN Response")

title(['Str=' num2str(ANN.Str) ...
       ' DelInp=' num2str(max(ANN.DelInp)) ...
       ' DelFe=' num2str(max(ANN.DelFe))]);

figure
plot(main.tr.perf)

title(['Str=' num2str(ANN.Str) ...
       ' DelInp=' num2str(max(ANN.DelInp)) ...
       ' DelFe=' num2str(max(ANN.DelFe))]);


end