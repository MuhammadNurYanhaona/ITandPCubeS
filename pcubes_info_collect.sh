#!/bin/bash

# this is a script that can be used to get information from a unix/linux system 
# about the hardware to construct its PCubeS description -- in addition a processor
# description file

printf "\nprocessor "
cat /proc/cpuinfo | grep "model name" | head -1
printf "total memory: "
memory=`cat /proc/meminfo | grep "MemTotal" | awk '{print $2}'`
memory_gb=$(($memory/(1024*1024)))
printf "$memory_gb GB\n"
echo "--------------------------------------------------------------------------"
cat /proc/cpuinfo | grep -e processor -e "physical id" -e "core id" | paste - - -
echo ""

echo "cache configuration"
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
	printf  "CPU: $cpu "
	echo "----------------------------------------"
	for cache in $cpu/cache/*; do
		printf "cache level: "
		cat $cache/level
		printf "type: "
		cat $cache/type
		printf "transaction width (bytes): "
		cat $cache/coherency_line_size
		printf "size: "
		cat $cache/size
		printf "participant CPUs: "
		cat $cache/shared_cpu_list
		printf "\n"
	done
done
