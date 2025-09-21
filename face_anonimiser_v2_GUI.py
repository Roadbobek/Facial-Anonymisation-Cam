from pydoc import stripid

import customtkinter as ctk


class face_anon_gui:
    def __init__(self, shared_state):  # Receive the values
        # Define root widget (window) of ctk
        self.root = ctk.CTk()

        ctk.set_appearance_mode("dark")

        self.shared_state = shared_state

        self.output_to_vcam_var = ctk.BooleanVar(value=self.shared_state['output_to_vcam'])
        self.flip_cam_input_var = ctk.BooleanVar(value=self.shared_state['flip_cam_input'])
        self.flip_output_to_vcam_var = ctk.BooleanVar(value=self.shared_state['flip_output_to_vcam'])
        self.fps_counter_var = ctk.BooleanVar(value=self.shared_state['fps_counter'])
        self.use_cuda_gpu_var = ctk.BooleanVar(value=self.shared_state['use_cuda_gpu'])
        self.scale_output_to_var = self.shared_state['scale_output_to']
        # print(self.scale_output_to_var) # DEBUG

        self.scale_output_to_var_text = ctk.StringVar(value=(f"{self.scale_output_to_var[0]}, {self.scale_output_to_var[1]}"))
        # print(self.scale_output_to_var_text) # DEBUG
        # print(self.scale_output_to_var_text.get()) # DEBUG




        # self.fps_counter = fps_counter_value # Now GUI has its own regular bool copy
        # self.fps_counter_var = ctk.BooleanVar(value=self.fps_counter) # Create special Tk bool var


        # Define root node window size
        self.root.geometry("480x720")
        # Define root window title
        self.root.title("Roadbobek Cam")


        # Create a label in window
        self.title_label = ctk.CTkLabel(self.root, text="Roadbobek Cam", font=ctk.CTkFont(size=30, weight="bold"))
        # Pack the label (initialise it) and add padding
        self.title_label.pack(padx=10, pady=(30, 30))
        # # Initialise the label with the grid() method
        # self.title_label.grid(padx=10, pady=(30, 30))


        self.scrollable_frame = ctk.CTkScrollableFrame(self.root, width=440, height=575)
        self.scrollable_frame.pack()


        self.output_to_vcam = ctk.CTkCheckBox(self.scrollable_frame, text="Output to virtual camera", width=(440 * 0.99), height=40, variable=self.output_to_vcam_var, command=self.output_to_vcam_func, onvalue=True, offvalue=False)
        self.output_to_vcam.pack(padx=15, pady=(10, 5))

        self.flip_cam_input_toggle = ctk.CTkCheckBox(self.scrollable_frame, text="Flip camera input", width=(440 * 0.99), height=40, variable=self.flip_cam_input_var, command=self.flip_cam_input_func, onvalue=True, offvalue=False)
        self.flip_cam_input_toggle.pack(padx=15, pady=(5, 5))

        self.flip_output_to_vcam_toggle = ctk.CTkCheckBox(self.scrollable_frame, text="Flip virtual camera output", width=(440 * 0.99), height=40, variable=self.flip_output_to_vcam_var, command=self.flip_output_to_vcam_func, onvalue=True, offvalue=False)
        self.flip_output_to_vcam_toggle.pack(padx=15, pady=(5, 5))

        self.fps_counter_toggle = ctk.CTkCheckBox(self.scrollable_frame, text="FPS Counter", width=(440 * 0.99), height=40, variable=self.fps_counter_var, command=self.fps_counter_func, onvalue=True, offvalue=False)
        self.fps_counter_toggle.pack(padx=15, pady=(5, 5))
        # fps_toggle = ctk.CTkCheckBox(scrollable_frame, width=(440 * 0.99), height=40)

        self.use_cuda_gpu_toggle = ctk.CTkCheckBox(self.scrollable_frame, text="Use CUDA GPU", width=(440 * 0.99), height=40, variable=self.use_cuda_gpu_var, command=self.use_cuda_gpu_func, onvalue=True, offvalue=False)
        self.use_cuda_gpu_toggle.pack(padx=15, pady=(5, 5))


        self.scale_output_to_frame = ctk.CTkFrame(self.scrollable_frame, width=440, height=40)


        self.scale_output_to_entry = ctk.CTkEntry(self.scale_output_to_frame, textvariable=self.scale_output_to_var_text, height=40, width=(440 * 0.4))
        self.scale_output_to_entry.pack(padx=10, pady=(10, 5), anchor='w', side='left')

        self.scale_output_to_apply = ctk.CTkButton(self.scale_output_to_frame, text="Apply", height=40, width=(440 * 0.2), command=self.scale_output_to_func)
        self.scale_output_to_apply.pack(padx=10, pady=(10, 5), side='right')


        self.scale_output_to_frame.pack()

        # self.checkbox_1 = ctk.CTkCheckBox(self.scale_output_to_frame, text="checkbox 1")
        # self.checkbox_1.pack(padx=10, pady=(10, 5), side='right')


        # Start ctk
        self.root.mainloop()



    def fps_counter_func(self):
        # print(self.fps_counter_var.get()) # DEBUG
        self.shared_state['fps_counter'] = self.fps_counter_var.get() # Update shared dict
        print(f"FPS Counter: {self.shared_state['fps_counter']}")

    def output_to_vcam_func(self):
        # print(self.output_to_vcam_var.get()) # DEBUG
        self.shared_state['output_to_vcam'] = self.output_to_vcam_var.get()
        print(f"Output to virtual camera: {self.shared_state['output_to_vcam']}")

    def flip_cam_input_func(self):
        # print(self.output_to_vcam_var.get()) # DEBUG
        self.shared_state['flip_cam_input'] = self.flip_cam_input_var.get()
        print(f"Flip camera input: {self.shared_state['flip_cam_input']}")

    def flip_output_to_vcam_func(self):
        # print(self.flip_output_to_vcam_var.get()) # DEBUG
        self.shared_state['flip_output_to_vcam'] = self.flip_output_to_vcam_var.get()
        print(f"Flip virtual camera output: {self.shared_state['flip_output_to_vcam']}")

    def use_cuda_gpu_func(self):
        # print(self.use_cuda_gpu_var.get()) # DEBUG
        self.shared_state['use_cuda_gpu'] = self.use_cuda_gpu_var.get()
        print(f"Use CUDA GPU: {self.shared_state['use_cuda_gpu']}")

    def scale_output_to_func(self):
        # print(self.scale_output_to_var) # DEBUG
        # print(self.scale_output_to_var_text) # DEBUG
        # print(self.scale_output_to_var_text.get()) # DEBUG

        # Split the string and convert to integers in one step
        self.scale_output_to_var = list(map(int, self.scale_output_to_var_text.get().split(", ")))

        # Make each number in the list positive (absolute)
        self.scale_output_to_var = [abs(num) for num in self.scale_output_to_var]
        # for i in range(len(self.scale_output_to_var)):
        #     self.scale_output_to_var[i] = abs(self.scale_output_to_var[i])

        # self.scale_output_to_var = self.scale_output_to_var_text.get().split(", ") # wtf was i thinking when i wrote this?
        # for i in self.scale_output_to_var:
        #     self.scale_output_to_var.append(int(i))

        # print(type(self.scale_output_to_var)) # DEBUG
        # print(self.scale_output_to_var) # DEBUG

        self.shared_state['scale_output_to'] = self.scale_output_to_var
        print(f"Scale output to: {self.shared_state['scale_output_to']}")
