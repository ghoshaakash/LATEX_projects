%Closing everything for next function call
close ALL;
clear ALL;


%data Prep of IRIS data
load("irisdata.mat")
data(:,5)=data(:,5)+1;


%Function Call
disp("IRIS data LOO rate:")
disp(LOO_CV(data));

%Closing everything for next function call
close ALL;
clear ALL;

%Importing Wine dataset

load("wine_dataset.mat")

labels=wineTargets.';
labelsRow=labels(:,1)+labels(:,2)*2+labels(:,3)*3;
data=horzcat(wineInputs.',labelsRow);


%Function Call
disp("Wine data LOO rate:")
disp(LOO_CV(data));

function rate=LOO_CV(data)

%clearing
close ALL;
clear ALL;
%getting rows columns and classes 
[r,c] = size(data);
[class,~]=size(unique(data(:,c)));

%We set the arr_rate[i] to the error on classifying data[i] as Cross
%checking data. It is 1 for proper classification, 0 otherwise

arr_rate=zeros(r,1);

for i=1:r
    arr_g=zeros(class,1);
    LOO_data=data([1:i-1,i+1:r],:);%Creating Dset without i^{th} row
    for j=1:class
       
        lastColumn = LOO_data(:, c);
        LOO_data_class=LOO_data(lastColumn==j,:);
        mu=mean(LOO_data_class(:,1:c-1));
        Sigma=cov(LOO_data_class(:,1:c-1));
        prior=size(LOO_data_class,1);
        arr_g(j)=mvnpdf(data(i,1:c-1),mu,Sigma)*prior;
        
    end
    [~,I]=max(arr_g);
    if I==data(i,c)
        arr_rate(i)=1;
    else 
        arr_rate(i)=0;
    end
    
end
rate=sum(arr_rate)/r;
end