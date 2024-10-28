# Final Annotator Script
# Purpose: To take the now-annotated csv file of 100 datapoints, and resolve any disagreement between annotators
#10/25/2024


#Load in Compiled Annotations(annotated_instances.csv

#Split information into 3 matrices; the input, annotator 1, and annotator 2
#Initialize an output matrix that correct annotations will be stored in

#Iterate through each datapoint
    #If annotators are in agreement for the final binary
        #Append the features and the final binary target label to the output matrix

    #Else If one annotator says yes and the other says no:
        #Retrive and display the relevant title and text for that annotation

        #Have the program state "Annotator x stated there was no violation, while Annotator y stated it was, for these reasons:"
        #Retrieve the target labels that annotator y marked as "yes", and display as a list

        #Ask the user to select which of the two annotators is correct
        #Append the features and correct annotator final binary label to output matrix

#Write the output matrix object to an output .csv file