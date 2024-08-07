# import tensorflow as tf
# from tensorboard.backend.event_processing import event_accumulator
# import csv

# # # Path to your event file
# # event_file = '/home/setareh/alltogether/deepreach/logs/Aug2_small_experiment/Aug2_160_logs/fourier/version_0/events.out.tfevents.1722610678.zhanggroup-station1.915057.0'
# event_file = '/home/setareh/alltogether/deepreach/logs/Aug6/Aug_6_fourier_num_terms=1/fourier/version_0/events.out.tfevents.1722963954.zhanggroup-station1.1016790.0'

# # Load the event file
# ea = event_accumulator.EventAccumulator(event_file)
# ea.Reload()

# # Get all available tags
# available_tags = ea.Tags()['scalars']
# print("Available tags:", available_tags)

# # Function to save data for a specific tag in a readable format
# def save_tag_to_readable_format(tag):
#     scalar_events = ea.Scalars(tag)
    
#     output_file = f'{tag}_readable_output_Aug6.csv'
#     with open(output_file, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["Epoch", "Value"])  # CSV header
        
#         # Get the maximum epoch number
#         max_epoch = max(event.step for event in scalar_events)
        
#         # Create a dictionary to store values for each epoch
#         epoch_values = {event.step: event.value for event in scalar_events}
        
#         # Write data for each epoch, using the last known value for missing epochs
#         last_known_value = None
#         for epoch in range(max_epoch + 1):
#             if epoch in epoch_values:
#                 value = epoch_values[epoch]
#                 last_known_value = value
#             elif last_known_value is not None:
#                 value = last_known_value
#             else:
#                 value = "N/A"  # For epochs before the first logged value
            
#             writer.writerow([epoch, f"{value:.4f}" if isinstance(value, float) else value])
    
#     print(f"Readable data for tag '{tag}' saved to {output_file}")

# # Specify the tags you want to save
# tags_to_save = ['train_loss_epoch', 'dirichlet_loss', 'diff_constraint_hom_loss']  # Add more tags as needed

# for tag in tags_to_save:
#     if tag in available_tags:
#         save_tag_to_readable_format(tag)
#     else:
#         print(f"Tag '{tag}' not found in the event file.")

# print("Data extraction and formatting complete.")


import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import csv

event_file = '/home/setareh/alltogether/deepreach/logs/Aug6/Aug_6_fourier_num_terms=1/fourier/version_0/events.out.tfevents.1722963954.zhanggroup-station1.1016790.0'

# Load the event file
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()

# Get all available tags
available_tags = ea.Tags()['scalars']
print("Available tags:", available_tags)

# Function to save data for a specific tag in a readable format
def save_tag_to_readable_format(tag):
    scalar_events = ea.Scalars(tag)
    
    output_file = f'{tag}_readable_output_Aug6.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Value"])  # CSV header
        
        # Write data for each logged epoch
        for event in scalar_events:
            writer.writerow([event.step, f"{event.value:.4f}"])
    
    print(f"Readable data for tag '{tag}' saved to {output_file}")

# Specify the tags you want to save
tags_to_save = ['train_loss_epoch', 'dirichlet_loss', 'diff_constraint_hom_loss']  # Add more tags as needed

for tag in tags_to_save:
    if tag in available_tags:
        save_tag_to_readable_format(tag)
    else:
        print(f"Tag '{tag}' not found in the event file.")

print("Data extraction and formatting complete.")