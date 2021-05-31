package gltest

import "github.com/whipstein/golinalg/mat"

type Memory struct {
	Infoc struct {
		Infot int
		Ok    bool
		Lerr  bool
	}
	Srnamc struct {
		Srnamt string
	}
	Sslct struct {
		Selopt int
		Seldim int
		Selval []bool
		Selwr  *mat.Vector
		Selwi  *mat.Vector
	}
	Claenv struct {
		Iparms []int
	}
	Mn struct {
		M      int
		N      int
		Mplusn int
		K      int
		I      int
		Fs     bool
	}
	Cenvir struct {
		Nproc  int
		Nshift int
		Maxb   int
	}
}

var Common Memory
