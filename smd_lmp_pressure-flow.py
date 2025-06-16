from mpi4py import MPI
from lammps import lammps
import os
import numpy as np
import pandas as pd
import random
import argparse
import time

class SMD:
    def __init__(self,inputfile,outputfile,result_dir,Ncell,param,pre_outputfile=None):
        self.inputlmp = inputfile
        self.output = outputfile
        self.result_dir = result_dir
        self.Ncell = int(Ncell)
        self.param = param
        self.pre_output = pre_outputfile
    
    def set_param(self,forceflag=False):
        df=pd.read_csv(self.param,header=None,delim_whitespace=True,index_col=0)
        
        self.Nrun = int(df.at["Nrun", 1])
        self.run_interval = int(df.at["sync_interval_step", 1])
        self.restart_step = int(df.at["restart_step", 1])
        self.timestep = float(df.at["timestep", 1])
        self.dy = float(df.at["dy", 1])
        self.density= float(df.at["density", 1])
        self.thermal_conductivity = float(df.at["thermal_conductivity", 1])
        self.temp_init = float(df.at["temp_init", 1])
        self.temp_w_bottom = float(df.at["temp_w_bottom", 1])
        self.temp_w_top = float(df.at["temp_w_top", 1])
        self.stress_w_top = float(df.at["stress_w_top", 1])
        self.str_coefficient = float(df.at["str_coefficient", 1])
        if forceflag==True :
            self.force_external = float(df.at["force_external", 1] )
        self.input_shear_array=np.zeros(self.Ncell,dtype="float64")
        self.input_temp_array=np.zeros(self.Ncell,dtype="float64")
        self.velocity_array=np.zeros(self.Ncell,dtype="float64")
        self.output_stress_array=np.zeros(self.Ncell,dtype="float64")
        self.output_temp_array=np.zeros(self.Ncell,dtype="float64")
        self.output_qyy_array=np.zeros(self.Ncell,dtype="float64")
    
    def SetdiffCouetteParam(self):
        self.diff_stress_matrix=np.zeros((self.Ncell,self.Ncell))
        self.bc_stress_array=np.zeros(self.Ncell)
        self.bc_stress_array[self.Ncell-1]=2.*self.stress_w_top
        self.c_stress=1./self.density/self.dy
        self.diff_shear_matrix=np.zeros((self.Ncell,self.Ncell))
        self.bc_shear_array=np.zeros(self.Ncell)
        self.c_shear=1./self.dy
        self.diff_temp_matrix=np.zeros((self.Ncell,self.Ncell))
        self.bc_temp_array=np.zeros(self.Ncell)
        self.bc_temp_array[0]=2.*self.temp_w_bottom
        self.bc_temp_array[self.Ncell-1]=2.*self.temp_w_top
        self.c_temp=2./3.*self.thermal_conductivity/self.density/self.dy/self.dy
       
        for i in range(0,self.Ncell):
            if i==self.Ncell-1:
                self.diff_stress_matrix[i,i]=-2.
            else:
                self.diff_stress_matrix[i,i]=-1.
                self.diff_stress_matrix[i,i+1]=1.
         
        for i in range(0,self.Ncell):
            if i==0:
                self.diff_shear_matrix[i,i]=1.
            else:
                self.diff_shear_matrix[i,i]=1.
                self.diff_shear_matrix[i,i-1]=-1.

        for i in range(0,self.Ncell):
            if i==0:
                self.diff_temp_matrix[i,i]=-3.
                self.diff_temp_matrix[i,i+1]=1.
            elif i==self.Ncell-1:
                self.diff_temp_matrix[i,i]=-3.
                self.diff_temp_matrix[i,i-1]=1.
            else:
                self.diff_temp_matrix[i,i]=-2.
                self.diff_temp_matrix[i,i-1]=1.
                self.diff_temp_matrix[i,i+1]=1.

    def set_diff_poiseuille_param(self):
        self.diff_stress_matrix=np.zeros((self.Ncell,self.Ncell))
        self.bc_stress_array=np.zeros(self.Ncell)
        self.bc_stress_array[self.Ncell-1]=2.*self.stress_w_top
        self.c_stress=1./self.density/self.dy
        self.diff_shear_matrix=np.zeros((self.Ncell,self.Ncell))
        self.bc_shear_array=np.zeros(self.Ncell)
        self.c_shear=1./self.dy
        self.diff_temp_matrix=np.zeros((self.Ncell,self.Ncell))
        self.bc_temp_array=np.zeros(self.Ncell)
        self.bc_temp_array[0]=2.*self.temp_w_bottom
        self.bc_temp_array[self.Ncell-1]=0.0
        self.c_temp=2./3.*self.thermal_conductivity/self.density/self.dy/self.dy
       
        for i in range(0,self.Ncell):
            if i==self.Ncell-1:
                self.diff_stress_matrix[i,i]=-2.
            else:
                self.diff_stress_matrix[i,i]=-1.
                self.diff_stress_matrix[i,i+1]=1.
       
        for i in range(0,self.Ncell):
            if i==0:
                self.diff_shear_matrix[i,i]=1.
            else:
                self.diff_shear_matrix[i,i]=1.
                self.diff_shear_matrix[i,i-1]=-1.

        for i in range(0,self.Ncell):
            if i==0:
                self.diff_temp_matrix[i,i]=-3.
                self.diff_temp_matrix[i,i+1]=1.
            elif i==self.Ncell-1:
                self.diff_temp_matrix[i,i]=-1.
                self.diff_temp_matrix[i,i-1]=1.
            else:
                self.diff_temp_matrix[i,i]=-2.
                self.diff_temp_matrix[i,i-1]=1.
                self.diff_temp_matrix[i,i+1]=1.

    def update_str(self,qyy_array):
        qeq=1./3.
        # Get the length of the input array
        n = len(qyy_array)
        # Initialize the output array
        b = np.zeros(n)
        # Special case: calculate b[0]
        b[0] = (3 * qyy_array[0] - qyy_array[1]) / 2
        # General case: calculate b[i]
        for i in range(1, n):
            b[i] = (qyy_array[i] + qyy_array[i - 1]) / 2
        #set thermal conductivity using STR normalized by equilib thermal conductivity
        #(th_yy-th_eq)/th_eq=Cb*(Qyy-Qeq)
        #th_yy/th_eq=1+Cb**(Qyy-Qeq)
        coefficient_thermal_array = 1.0 + self.str_coefficient*(b-qeq)
        #Update bc_temp_array and diff temp matrix
        self.bc_temp_array[0]=2.*self.temp_w_bottom*coefficient_thermal_array[0]
        for i in range(0,self.Ncell):
            if i==0:
                self.diff_temp_matrix[i,i]=-(2*coefficient_thermal_array[0]+coefficient_thermal_array[1])
                self.diff_temp_matrix[i,i+1]=coefficient_thermal_array[1]
            elif i==self.Ncell-1:
                self.diff_temp_matrix[i,i]=-coefficient_thermal_array[self.Ncell-1]
                self.diff_temp_matrix[i,i-1]=coefficient_thermal_array[self.Ncell-1]
            else:
                self.diff_temp_matrix[i,i]=-(coefficient_thermal_array[i-1]+coefficient_thermal_array[i+1])
                self.diff_temp_matrix[i,i-1]=coefficient_thermal_array[i-1]
                self.diff_temp_matrix[i,i+1]=coefficient_thermal_array[i+1]
          
        #with open("STR.log","a") as ft:
        #    print("qyy",qyy_array,file=ft)
        #    print("qyy_1/2",b,file=ft)
        #    print("th",coefficient_thermal_array,file=ft)
        #    print("th_bc",self.bc_temp_array,file=ft)
        #    print("th_matrix",self.diff_temp_matrix,file=ft)
     
    @staticmethod
    def update_diff_method(input_array,diff_matrix,bc_array,constant):
        output_array=constant*(np.dot(diff_matrix,input_array)+bc_array)
        return output_array

    @staticmethod
    def compress_array(original_array, multiple):
        indices_multiple = np.arange(0, len(original_array), multiple)
        new_array = np.array(original_array)[indices_multiple]
        return new_array
    
    @staticmethod
    def expand_array(original_array, factor):
        expanded_array = np.empty(len(original_array) * factor, dtype=original_array.dtype)
        for i, value in enumerate(original_array):
            expanded_array[i * factor:(i + 1) * factor] = value
        return expanded_array
    
    @staticmethod
    def get_bond_orientation_tensor(bond_array):
        bmin=0.97
        x = bond_array[:, 0]
        y = bond_array[:, 1]
        z = bond_array[:, 2]

        b = np.column_stack((x**2, y**2, z**2, x*y, x*z, y*z))
    
        column_means = np.mean(b, axis=0)/bmin/bmin
        return column_means[1],column_means[3]


    def initialize_array(self):
      if self.pre_output ==None:
        for i in range(self.Ncell):
          self.input_shear_array[i] = 0.0
          self.input_temp_array[i] = self.temp_init 
          self.velocity_array[i] = 0.0
          self.output_stress_array[i] = 0.0
          self.output_temp_array[i] = 0.0
          output=os.path.join(self.result_dir,"cell_"+str(i),self.output)
          #write header output files
      else:
        for i in range(self.Ncell):
          pre_result_path=os.path.join(self.result_dir,"cell_"+str(i),self.pre_output)
          print(pre_result_path,i) 
          with open(pre_result_path) as fi:
            last_property=fi.readlines()[-1].split()
            last_inputtemp= float(last_property[2])
            last_velocity= float(last_property[3])
            last_shear= float(last_property[4])
          self.input_temp_array[i] = last_inputtemp
          self.velocity_array[i] = last_velocity
          self.input_shear_array[i] = last_shear
          self.output_stress_array[i] = 0.0
          self.output_temp_array[i] = 0.0

    def initialize_scatter_gather_array(self,size,md_size):
        self.scatter_gather_velocity_array =np.zeros(size)
        self.scatter_gather_input_temp_array =np.zeros(size)
        self.scatter_gather_input_shear_array =np.zeros(size)
        self.scatter_gather_output_temp_array =np.zeros(size)
        self.scatter_gather_output_stress_array =np.zeros(size)
        self.scatter_gather_output_qyy_array =np.zeros(size)

        if self.pre_output ==None:
            for i in range(self.Ncell):
                for j in range(md_size):
                    self.scatter_gather_input_temp_array[i*md_size+j] = self.temp_init 
        else:
            for i in range(self.Ncell):
                pre_result_path=os.path.join(self.result_dir,"cell_"+str(i),self.pre_output)
                print(pre_result_path,i) 
                with open(pre_result_path) as fi:
                    last_property=fi.readlines()[-1].split()
                    last_inputtemp= float(last_property[2])
                    last_velocity= float(last_property[3])
                    last_shear= float(last_property[4])
                for j in range(md_size):
                    self.scatter_gather_input_temp_array[i*md_size+j] = last_inputtemp
                    self.scatter_gather_velocity_array[i*md_size+j] = last_velocity
                    self.scatter_gather_input_shear_array[i*md_size+j] = last_shear
  
    def setup_simulation(self):
        self.set_param(True)
        self.set_diff_poiseuille_param()
        self.initialize_array()
        self.cfd_comm = MPI.COMM_WORLD
        self.rank = self.cfd_comm.Get_rank()
        self.size = self.cfd_comm.Get_size()

        self.color = self.rank // (self.size // self.Ncell)
        self.md_comm = self.cfd_comm.Split(self.color)
        self.md_rank = self.md_comm.Get_rank()
        self.md_size = self.md_comm.Get_size()

        self.initialize_scatter_gather_array(self.size, self.md_size)

        self.recv_velocity = np.empty(1, dtype=np.float64)
        self.recv_input_shear = np.empty(1, dtype=np.float64)
        self.recv_input_temp = np.empty(1, dtype=np.float64)
        self.recv_output_stress = np.empty(1, dtype=np.float64)
        self.recv_output_temp = np.empty(1, dtype=np.float64)
        self.recv_output_qyy = np.empty(1, dtype=np.float64)

        cell = os.path.join(self.result_dir, f"cell_{self.color}")
        os.chdir(cell)

        if self.md_rank==0:
            #with open("time.log","w") as ft:
            #    print("Total","Scatter","Pre","LAMMPS","Post","Gather",file=ft)
            with open(self.output, mode='w') as fo:
                print("time","press","temp","velocity","shear","intstress","inttemp","Qyy","Qxy",file=fo,sep=" ")

        self.lmp = lammps(comm=self.md_comm)
        self.lmp.file(self.inputlmp)
        self.lmp.command(f"timestep {self.timestep}")
        self.lmp.command("compute bondsmd all bond/local dx dy dz")
        self.lmp.command(f"thermo {self.run_interval}")

    def finalize_simulation(self):
        self.lmp.command("write_data data.last pair ij")
        self.lmp.command("write_restart restart.last")
        MPI.Finalize()       

    def get_bond(self):

        bond_array = self.lmp.numpy.extract_compute("bondsmd", 2, 2)
        rows, cols = bond_array.shape  

        sendcount = np.array(rows * cols, dtype=np.int32)
        recvcounts = self.md_comm.gather(sendcount, root=0)

        if self.md_rank == 0:
            recvcounts = np.array(recvcounts, dtype=np.int32)
            displs = np.insert(np.cumsum(recvcounts[:-1]), 0, 0).astype(np.int32)
            total_size = np.sum(recvcounts)
            recvbuf = np.empty(total_size, dtype=np.float64)
        else:
            recvbuf = None
            displs = None

        self.md_comm.Gatherv(sendbuf=bond_array.flatten(),
                             recvbuf=(recvbuf, recvcounts, displs, MPI.DOUBLE),
                             root=0)

        if self.md_rank == 0:
            total_rows = total_size // cols  
            gathered_array = recvbuf.reshape((total_rows, cols))
            self.qyy, self.qxy = self.get_bond_orientation_tensor(gathered_array)

        else:
            self.qyy = None    

    def run_simulation(self):

        for N in range(self.Nrun):
            
            self.scatter_gather_velocity_array = self.expand_array(self.velocity_array, self.md_size)
            self.scatter_gather_input_shear_array = self.expand_array(self.input_shear_array, self.md_size)
            self.scatter_gather_input_temp_array = self.expand_array(self.input_temp_array, self.md_size)

            self.cfd_comm.Scatter(self.scatter_gather_velocity_array, self.recv_velocity, root=0)
            self.cfd_comm.Scatter(self.scatter_gather_input_shear_array, self.recv_input_shear, root=0)
            self.cfd_comm.Scatter(self.scatter_gather_input_temp_array, self.recv_input_temp, root=0)
            self.cfd_comm.Scatter(self.scatter_gather_output_stress_array, self.recv_output_stress, root=0)
            self.cfd_comm.Scatter(self.scatter_gather_output_temp_array, self.recv_output_temp, root=0)
           
            self.lmp.command(f"fix sllod all nvt/sllod temp {self.recv_input_temp[0]} {self.recv_input_temp[0]} 999999 tchain 1")
            self.lmp.command(f"fix deform all deform 1 xy erate {self.recv_input_shear[0]} remap v")
            self.lmp.command(f"velocity all scale {self.recv_input_temp[0]}")
            self.lmp.command("compute temp_deform all temp/deform")
            self.lmp.command("compute mystress all pressure temp_deform")
            self.lmp.command("fix ave all ave/time 1 400 400 c_temp_deform c_mystress[*] file smd.profile")
            self.lmp.command(f"variable integrated_T equal f_ave[1]*{self.run_interval}*{self.timestep}")
            self.lmp.command(f"variable integrated_pxy equal -1*f_ave[5]*{self.run_interval}*{self.timestep}")
            self.lmp.command(f"variable shear_velocity atom vx+y*{self.recv_input_shear[0]}")
            self.lmp.command("velocity all set v_shear_velocity NULL NULL")
            self.lmp.command(f"run {self.run_interval}")
  
            MDtime = self.lmp.get_thermo("time")
            press = self.lmp.get_thermo("press")
            flow_temp=self.lmp.extract_compute("temp_deform",0,0)
            integrated_temp = self.lmp.extract_variable("integrated_T",None,0) 
            integrated_stress = self.lmp.extract_variable("integrated_pxy",None,0) 
            
            if N % self.restart_step == 0:
                self.lmp.command("write_data data.tmp pair ij")
                self.lmp.command("write_restart restart.tmp")
    
#get bond orientatin
            self.get_bond()

#write output 
            if self.md_rank==0:
                self.recv_input_temp[0] = flow_temp
                self.recv_output_temp[0] = integrated_temp
                self.recv_output_stress[0] = integrated_stress
                self.recv_output_qyy[0] = self.qyy
                with open(self.output, mode='a') as fo:
                  print(MDtime,press,self.recv_input_temp[0],self.recv_velocity[0],self.recv_input_shear[0],self.recv_output_stress[0],self.recv_output_temp[0],self.recv_output_qyy[0],self.qxy,file=fo,sep=" ")
            else:
                self.recv_input_temp[0] = None
                self.recv_output_temp[0] = None
                self.recv_output_stress[0] = None
                self.recv_output_qyy[0] = None

#stop shear 
            self.lmp.command("variable integrated_T delete")
            self.lmp.command("variable integrated_pxy delete")
            self.lmp.command("unfix ave")
            self.lmp.command("uncompute mystress")
            self.lmp.command("uncompute temp_deform")
            self.lmp.command("unfix deform")
            self.lmp.command("unfix sllod")
            self.lmp.command(f"variable shear_velocity atom vx-y*{self.recv_input_shear[0]}")
            self.lmp.command("velocity all set v_shear_velocity NULL NULL")

            self.cfd_comm.Gather(sendbuf=self.recv_velocity, recvbuf=self.scatter_gather_velocity_array,root=0)
            self.cfd_comm.Gather(sendbuf=self.recv_input_shear,recvbuf=self.scatter_gather_input_shear_array,root=0)
            self.cfd_comm.Gather(sendbuf=self.recv_input_temp,recvbuf=self.scatter_gather_input_temp_array,root=0)
            self.cfd_comm.Gather(sendbuf=self.recv_output_temp,recvbuf=self.scatter_gather_output_temp_array,root=0)
            self.cfd_comm.Gather(sendbuf=self.recv_output_stress,recvbuf=self.scatter_gather_output_stress_array,root=0)
            self.cfd_comm.Gather(sendbuf=self.recv_output_qyy,recvbuf=self.scatter_gather_output_qyy_array,root=0)
  
  #Gend=time.time()
  #Gather_time=Gend-Gstart
  #Tend=time.time()
  #T_time=Tend-Tstart
  #with open("time.log","a") as ft:
  #  print(T_time,Pre_time,Run_time,Post_time,Gather_time,file=ft)

# sync and update temperature and shear
            if self.rank==0:
              ##compress array
                self.velocity_array = self.compress_array(self.scatter_gather_velocity_array,self.md_size)
                self.input_temp_array = self.compress_array(self.scatter_gather_input_temp_array,self.md_size)
                self.output_stress_array = self.compress_array(self.scatter_gather_output_stress_array,self.md_size)
                self.output_temp_array = self.compress_array(self.scatter_gather_output_temp_array,self.md_size)
                self.output_qyy_array = self.compress_array(self.scatter_gather_output_qyy_array,self.md_size)
              
                self.update_str(self.output_qyy_array)
              
                self.velocity_array=self.velocity_array+self.update_diff_method(self.output_stress_array,self.diff_stress_matrix,self.bc_stress_array,self.c_stress)
              ##Poiseuille
                self.velocity_array=self.velocity_array+self.force_external
                self.input_shear_array=self.update_diff_method(self.velocity_array,self.diff_shear_matrix,self.bc_shear_array,self.c_shear)
                self.input_temp_array=self.input_temp_array+self.update_diff_method(self.output_temp_array,self.diff_temp_matrix,self.bc_temp_array,self.c_temp)
 
            self.cfd_comm.barrier()
    #with open("debug.log","a") as f:
    #  print("before",file=f)
    #  print("output stress",SMD.output_stress_array,file=f)
    #  print("velocity",SMD.velocity_array,file=f)
    #  print("shear",SMD.input_shear_array,file=f)
    #  print("input temp",SMD.input_temp_array,file=f)
    #  print("output temp",SMD.output_temp_array,file=f)
    #  print("force_external",SMD.force_external,file=f)
    #  print(" ",file=f)

    #with open("scatter_gather.log","a") as f:
    #  print("before",file=f)
    #  print("output stress",SMD.scatter_gather_output_stress_array,file=f)
    #  print("velocity",SMD.scatter_gather_velocity_array,file=f)
    #  print("shear",SMD.scatter_gather_input_shear_array,file=f)
    #  print("input temp",SMD.scatter_gather_input_temp_array,file=f)
    #  print("output temp",SMD.scatter_gather_output_temp_array,file=f)
    #  print("force_external",SMD.force_external,file=f)
    #  print(" ",file=f)
    
def main():

    parser = argparse.ArgumentParser(description="smd to simulate lubrication flow")
    parser.add_argument("-i",'--input', help="input lammps script",required=True)
    parser.add_argument('-o',"--output",help="output file to write results",required=True)
    parser.add_argument('-d',"--dir", help="directory to read input and dump outout",required=True)
    parser.add_argument('-n',"--ncell", help="the number of cell",required=True)
    parser.add_argument('-p',"--param", help="SMD parameter file path",required=True)
    parser.add_argument('-r',"--restart", help="previous output file to restart simulation")
    args = parser.parse_args()

    smd=SMD(args.input,args.output,args.dir,args.ncell,args.param,args.restart)
    smd.setup_simulation()
    smd.run_simulation()
    smd.finalize_simulation()

if __name__ == "__main__":
    main()


