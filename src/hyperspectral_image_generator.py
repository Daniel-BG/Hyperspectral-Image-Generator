#level.
# setBlock(x,y,z,id)
# setBlockDataAt(x,y,z,data)
# setBlock(x,y,z,id,data)
# blockAt(x, y, z)
# blockDataAt(x, y, z)
import numpy
import time
import math
from array import array
from materials import *

# opciones del programa
displayName = "Hyperspectral image generator"
inputs = (
  ("Number of bands", (10, 1, 300)),  #Integer input, default: 10, min: 1, max: 300.
  ("Wavelength minimum", 410),
  ("Wavelength maximum", 2500),
  ("Output file", "string"),
  ("Sample every n blocks", (1,1,256)),
  ("Merge every n samples", (1,1,256)),
  ("Hyperspectral Image Generator", "label"),
)

class Constants:
	VERSION = "Hyperspectral_Image_Generator V0.1"
	LIGHT_THRESHOLD = 0.0001
	
	
CONST = Constants()
		
#https://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)
		
# From SethBling's StructureDeleter filter
# http://youtube.com/SethBling
########## Fast data access ##########
from pymclevel import ChunkNotPresent
GlobalChunkCache = {}
GlobalLevel = None

def getChunk(x, z):
	global GlobalChunkCache
	global GlobalLevel
	chunkCoords = (x>>4, z>>4)
	if chunkCoords not in GlobalChunkCache:
		try:
			GlobalChunkCache[chunkCoords] = GlobalLevel.getChunk(x>>4, z>>4)
		except ChunkNotPresent:
			return None
	return GlobalChunkCache[chunkCoords]
	
def blockAndDataAt(x,y,z):
	chunk = getChunk(x, z)
	if chunk == None:
		return 0
	return (chunk.Blocks[x%16][z%16][y], chunk.Data[x%16][z%16][y])
		
########## End fast data access ##########
		
def getDictId(block,data):
	return block + (data << 8)
				

def perform(level, box, options):
	filename = options["Output file"] + ".bsq"
	bands = options["Number of bands"]
	min_wav = options["Wavelength minimum"]
	max_wav = options["Wavelength maximum"]
	sample_step = options["Sample every n blocks"]
	merge_freq = options["Merge every n samples"]
	wav_step = (1.0 * (max_wav - min_wav)) / (bands - 1)
	global GlobalLevel
	GlobalLevel = level
	
	# intervals. Width and depth are rounded up to the nearest integer
	width = (box.maxx - box.minx + sample_step - 1)//sample_step
	depth = (box.maxz - box.minz + sample_step - 1)//sample_step
	
	# matrix which will contain the spectrum
	# specMatrix = [[[]]]
	# specMatrix = [[[0 for y in xrange(bands)] for z in xrange(depth)] for x in xrange(width)]
	specMatrix = numpy.zeros(shape=(bands,width,depth),dtype=numpy.dtype('f4'))
	# dictionary of spectrums
	materialDict = materials()
	
	# try to open the file before calculating the matrix just in case
	# the file cannot be opened afterwards and the calculations are lost
	file = open(filename, 'wb')
	file.truncate()
	
	# xrange returns values as needed instead of creating them all at once
	# loop on all selected coordinates
	
	with Timer('Processing time'):
		for x in xrange(box.minx,box.maxx,sample_step):
			for z in xrange(box.minz,box.maxz,sample_step):
				# light will be used to average values of the spectrum
				# of all visible objects within a column
				columnLight = 1.0
				# loop from top to bottom (direction of light)
				# when looping backwards substract 1 to the interval
				for y in xrange(box.maxy-1,box.miny-1, -1):
					(block,data) = blockAndDataAt(x,y,z)
					#block = level.blockAt(x,y,z)
					#data  = level.blockDataAt(x,y,z)
					# 8 bits per block and 4 per data
					type  = getDictId(block,data)
					# if we don't have a spectrum for the current material, 
					# ignore it and go to next
					if type not in materialDict:
						continue
					# get current material for block type
					currentMaterial = materialDict[type]
					# get light output of current material and update light to the next
					currentLight = columnLight * (1.0 - currentMaterial.transparency)
					columnLight *= currentMaterial.transparency
					# precalc index access
					cx = (x-box.minx)//sample_step
					cz = (z-box.minz)//sample_step
					currentWav = min_wav
					for k in xrange(bands):
						# update wavelenght on spectral matrix
						specMatrix[k][cx][cz] += currentLight*currentMaterial.get_reflectancie(currentWav)
						# update wavelengt for next step
						currentWav += wav_step
					# if there is a non significant amount of light left, continue onto next pixel
					if columnLight < CONST.LIGHT_THRESHOLD:
						continue
		
		# interpolate neighbours to reduce image size if desired
		if merge_freq > 1:
			specMatrix_temp = numpy.zeros(shape=(bands,
												(width+merge_freq-1)//merge_freq,
												(depth+merge_freq-1)//merge_freq),
										  dtype=numpy.dtype('f4'))
			# precalculate linear interpolation factor to speed up computation
			merge_factor = 1.0/(merge_freq*merge_freq)
			for x in xrange(width):
				for z in xrange(depth):
					for k in xrange(bands):
						specMatrix_temp[k][x//merge_freq][z//merge_freq] += merge_factor*specMatrix[k][x][z]
			specMatrix = specMatrix_temp
			# save new values to file
			width = (width+merge_freq-1)//merge_freq
			depth = (depth+merge_freq-1)//merge_freq
	
	# write data file
	specMatrix.tofile(file)
	file.close()
	
	# now write the header file
	filename = options["Output file"] + ".hdr"
	file = open(filename, 'w')
	file.truncate()
	file.write("ENVI\ndescription = {Generated using "+ CONST.VERSION + "}\n")
	file.write("samples = " + str(depth) + "\n")
	file.write("lines = " + str(width) + "\n")
	file.write("bands = " + str(bands) + "\n")
	file.write("header offset = 0\nfile type = ENVI Standard\ndata type = 4\ninterleave = bsq\nsensor type = Unknown\nbyte order = 0\ninterleave = bsq\n")
	file.write("wavelength = {\n")
	current = min_wav
	for i in xrange(bands):
		file.write(str(current))
		if i != bands - 1:
			file.write(", ")
		current += wav_step
	file.write("}")
	
	file.close()
	
	
	