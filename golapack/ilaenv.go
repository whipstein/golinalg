package golapack

import "bytes"

// Ilaenv is called from the LAPACK routines to choose problem-dependent
// parameters for the local environment.  See ISPEC for a description of
// the parameters.
//
// ILAENV returns an INTEGER
// if ILAENV >= 0: ILAENV returns the value of the parameter specified by ISPEC
// if ILAENV < 0:  if ILAENV = -k, the k-th argument had an illegal value.
//
// This version provides a set of parameters which should give good,
// but not optimal, performance on many of the currently available
// computers.  Users are encouraged to modify this subroutine to set
// the tuning parameters for their particular machine using the option
// and problem size information in the arguments.
//
// This routine will not function correctly if it is converted to all
// lower case.  Converting it to all upper case is allowed.
func Ilaenv(ispec int, name string, opts []byte, n1, n2, n3, n4 int) (ilaenvReturn int) {
	var cname, sname, twostage bool
	var c1 byte
	var c2, c3, c4 string
	var i, ic, iz, nb, nbmin, nx int
	subnam := make([]byte, 16)

	switch ispec {
	case 1:
		goto label10
	case 2:
		goto label10
	case 3:
		goto label10
	case 4:
		goto label80
	case 5:
		goto label90
	case 6:
		goto label100
	case 7:
		goto label110
	case 8:
		goto label120
	case 9:
		goto label130
	case 10:
		goto label140
	case 11:
		goto label150
	case 12:
		goto label160
	case 13:
		goto label160
	case 14:
		goto label160
	case 15:
		goto label160
	case 16:
		goto label160
	}

	//     Invalid value for ISPEC
	ilaenvReturn = -1
	return

label10:
	;

	//     Convert NAME to upper case if the first character is lower case.
	ilaenvReturn = 1
	subnam = []byte(name)
	ic = int(subnam[0])
	iz = int('Z')
	if iz == 90 || iz == 122 {
		//        ASCII character set
		if ic >= 97 && ic <= 122 {
			subnam[0] = byte(ic - 32)
			for i = 2; i <= 6; i++ {
				ic = int(subnam[i-1])
				if ic >= 97 && ic <= 122 {
					subnam[i-1] = byte(ic - 32)
				}
			}
		}

	} else if iz == 233 || iz == 169 {
		//        EBCDIC character set
		if (ic >= 129 && ic <= 137) || (ic >= 145 && ic <= 153) || (ic >= 162 && ic <= 169) {
			subnam[0] = byte(ic + 64)
			for i = 2; i <= 6; i++ {
				ic = int(subnam[i-1])
				if (ic >= 129 && ic <= 137) || (ic >= 145 && ic <= 153) || (ic >= 162 && ic <= 169) {
					subnam[i-1] = byte(ic + 64)
				}
			}
		}

	} else if iz == 218 || iz == 250 {
		//        Prime machines:  ASCII+128
		if ic >= 225 && ic <= 250 {
			subnam[0] = byte(ic - 32)
			for i = 2; i <= 6; i++ {
				ic = int(subnam[i-1])
				if ic >= 225 && ic <= 250 {
					subnam[i-1] = byte(ic - 32)
				}
			}
		}
	}

	c1 = name[0]
	sname = c1 == 'S' || c1 == 'D'
	cname = c1 == 'C' || c1 == 'Z'
	if !(cname || sname) {
		return
	}
	c2 = name[1:3]
	if len(name) > 6 {
		c3 = name[3:6]
		c4 = c3[1:3]
	} else {
		c3 = name[3:]
		c4 = c3[1:]
	}
	twostage = len(subnam) >= 11 && subnam[10] == '2'

	switch ispec {
	case 1:
		goto label50
	case 2:
		goto label60
	case 3:
		goto label70
	}

label50:
	;

	//     ISPEC = 1:  block size
	//
	//     In these examples, separate code is provided for setting NB for
	//     real and complex.  We assume that NB will take the same value in
	//     single or double precision.
	nb = 1

	if bytes.Equal(subnam[1:6], []byte("laorh")) {
		//        This is for *laorhR_GETRFNP routine
		if sname {
			nb = 32
		} else {
			nb = 32
		}
	} else if c2 == "ge" {
		if c3 == "trf" {
			if sname {
				nb = 64
			} else {
				nb = 64
			}
		} else if c3 == "qrf" || c3 == "rqf" || c3 == "lqf" || c3 == "qlf" {
			if sname {
				nb = 32
			} else {
				nb = 32
			}
		} else if c3 == "qr" {
			if n3 == 1 {
				if sname {
					//     M*N
					if (n1*n2 <= 131072) || (n1 <= 8192) {
						nb = n1
					} else {
						nb = 32768 / n2
					}
				} else {
					if (n1*n2 <= 131072) || (n1 <= 8192) {
						nb = n1
					} else {
						nb = 32768 / n2
					}
				}
			} else {
				if sname {
					nb = 1
				} else {
					nb = 1
				}
			}
		} else if c3 == "lq" {
			if n3 == 2 {
				if sname {
					//     M*N
					if (n1*n2 <= 131072) || (n1 <= 8192) {
						nb = n1
					} else {
						nb = 32768 / n2
					}
				} else {
					if (n1*n2 <= 131072) || (n1 <= 8192) {
						nb = n1
					} else {
						nb = 32768 / n2
					}
				}
			} else {
				if sname {
					nb = 1
				} else {
					nb = 1
				}
			}
		} else if c3 == "hrd" {
			if sname {
				nb = 32
			} else {
				nb = 32
			}
		} else if c3 == "brd" {
			if sname {
				nb = 32
			} else {
				nb = 32
			}
		} else if c3 == "tri" {
			if sname {
				nb = 64
			} else {
				nb = 64
			}
		}
	} else if c2 == "po" {
		if c3 == "trf" {
			if sname {
				nb = 64
			} else {
				nb = 64
			}
		}
	} else if c2 == "sy" {
		if c3 == "trf" {
			if sname {
				if twostage {
					nb = 192
				} else {
					nb = 64
				}
			} else {
				if twostage {
					nb = 192
				} else {
					nb = 64
				}
			}
		} else if sname && c3 == "trd" {
			nb = 32
		} else if sname && c3 == "gst" {
			nb = 64
		}
	} else if cname && c2 == "he" {
		if c3 == "trf" {
			if twostage {
				nb = 192
			} else {
				nb = 64
			}
		} else if c3 == "trd" {
			nb = 32
		} else if c3 == "gst" {
			nb = 64
		}
	} else if sname && c2 == "or" {
		if c3[0] == 'g' {
			if c4 == "qr" || c4 == "rq" || c4 == "lq" || c4 == "ql" || c4 == "hr" || c4 == "tr" || c4 == "br" {
				nb = 32
			}
		} else if c3[0] == 'm' {
			if c4 == "qr" || c4 == "rq" || c4 == "lq" || c4 == "ql" || c4 == "hr" || c4 == "tr" || c4 == "br" {
				nb = 32
			}
		}
	} else if cname && c2 == "un" {
		if c3[0] == 'g' {
			if c4 == "qr" || c4 == "rq" || c4 == "lq" || c4 == "ql" || c4 == "hr" || c4 == "tr" || c4 == "br" {
				nb = 32
			}
		} else if c3[0] == 'm' {
			if c4 == "qr" || c4 == "rq" || c4 == "lq" || c4 == "ql" || c4 == "hr" || c4 == "tr" || c4 == "br" {
				nb = 32
			}
		}
	} else if c2 == "gb" {
		if c3 == "trf" {
			if sname {
				if n4 <= 64 {
					nb = 1
				} else {
					nb = 32
				}
			} else {
				if n4 <= 64 {
					nb = 1
				} else {
					nb = 32
				}
			}
		}
	} else if c2 == "pb" {
		if c3 == "trf" {
			if sname {
				if n2 <= 64 {
					nb = 1
				} else {
					nb = 32
				}
			} else {
				if n2 <= 64 {
					nb = 1
				} else {
					nb = 32
				}
			}
		}
	} else if c2 == "tr" {
		if c3 == "tri" {
			if sname {
				nb = 64
			} else {
				nb = 64
			}
		} else if c3 == "evc" {
			if sname {
				nb = 64
			} else {
				nb = 64
			}
		}
	} else if c2 == "la" {
		if c3 == "uum" {
			if sname {
				nb = 64
			} else {
				nb = 64
			}
		}
	} else if sname && c2 == "st" {
		if c3 == "ebz" {
			nb = 1
		}
	} else if c2 == "gg" {
		nb = 32
		if c3 == "hd3" {
			if sname {
				nb = 32
			} else {
				nb = 32
			}
		}
	}
	ilaenvReturn = nb
	return

label60:
	;

	//     ISPEC = 2:  minimum block size
	nbmin = 2
	if c2 == "ge" {
		if c3 == "qrf" || c3 == "rqf" || c3 == "lqf" || c3 == "qlf" {
			if sname {
				nbmin = 2
			} else {
				nbmin = 2
			}
		} else if c3 == "hrd" {
			if sname {
				nbmin = 2
			} else {
				nbmin = 2
			}
		} else if c3 == "brd" {
			if sname {
				nbmin = 2
			} else {
				nbmin = 2
			}
		} else if c3 == "tri" {
			if sname {
				nbmin = 2
			} else {
				nbmin = 2
			}
		}
	} else if c2 == "sy" {
		if c3 == "trf" {
			if sname {
				nbmin = 8
			} else {
				nbmin = 8
			}
		} else if sname && c3 == "trd" {
			nbmin = 2
		}
	} else if cname && c2 == "he" {
		if c3 == "trd" {
			nbmin = 2
		}
	} else if sname && c2 == "or" {
		if c3[0] == 'g' {
			if c4 == "qr" || c4 == "rq" || c4 == "lq" || c4 == "ql" || c4 == "hr" || c4 == "tr" || c4 == "br" {
				nbmin = 2
			}
		} else if c3[0] == 'm' {
			if c4 == "qr" || c4 == "rq" || c4 == "lq" || c4 == "ql" || c4 == "hr" || c4 == "tr" || c4 == "br" {
				nbmin = 2
			}
		}
	} else if cname && c2 == "un" {
		if c3[0] == 'g' {
			if c4 == "qr" || c4 == "rq" || c4 == "lq" || c4 == "ql" || c4 == "hr" || c4 == "tr" || c4 == "br" {
				nbmin = 2
			}
		} else if c3[0] == 'm' {
			if c4 == "qr" || c4 == "rq" || c4 == "lq" || c4 == "ql" || c4 == "hr" || c4 == "tr" || c4 == "br" {
				nbmin = 2
			}
		}
	} else if c2 == "gg" {
		nbmin = 2
		if c3 == "hd3" {
			nbmin = 2
		}
	}
	ilaenvReturn = nbmin
	return

label70:
	;

	//     ISPEC = 3:  crossover point
	nx = 0
	if c2 == "ge" {
		if c3 == "qrf" || c3 == "rqf" || c3 == "lqf" || c3 == "qlf" {
			if sname {
				nx = 128
			} else {
				nx = 128
			}
		} else if c3 == "hrd" {
			if sname {
				nx = 128
			} else {
				nx = 128
			}
		} else if c3 == "brd" {
			if sname {
				nx = 128
			} else {
				nx = 128
			}
		}
	} else if c2 == "sy" {
		if sname && c3 == "trd" {
			nx = 32
		}
	} else if cname && c2 == "he" {
		if c3 == "trd" {
			nx = 32
		}
	} else if sname && c2 == "or" {
		if c3[0] == 'g' {
			if c4 == "qr" || c4 == "rq" || c4 == "lq" || c4 == "ql" || c4 == "hr" || c4 == "tr" || c4 == "br" {
				nx = 128
			}
		}
	} else if cname && c2 == "un" {
		if c3[0] == 'g' {
			if c4 == "qr" || c4 == "rq" || c4 == "lq" || c4 == "ql" || c4 == "hr" || c4 == "tr" || c4 == "br" {
				nx = 128
			}
		}
	} else if c2 == "gg" {
		nx = 128
		if c3 == "hd3" {
			nx = 128
		}
	}
	ilaenvReturn = nx
	return

label80:
	;

	//     ISPEC = 4:  number of shifts (used by xHSEQR)
	ilaenvReturn = 6
	return

label90:
	;

	//     ISPEC = 5:  minimum column dimension (not used)
	ilaenvReturn = 2
	return

label100:
	;

	//     ISPEC = 6:  crossover point for SVD (used by xGELSS and xGESVD)
	ilaenvReturn = int(float64(min(n1, n2)) * 1.6)
	return

label110:
	;

	//     ISPEC = 7:  number of processors (not used)
	ilaenvReturn = 1
	return

label120:
	;

	//     ISPEC = 8:  crossover point for multishift (used by xHSEQR)
	ilaenvReturn = 50
	return

label130:
	;

	//     ISPEC = 9:  maximum size of the subproblems at the bottom of the
	//                 computation tree in the divide-and-conquer algorithm
	//                 (used by xGELSD and xGESDD)
	ilaenvReturn = 25
	return

label140:
	;

	//     ISPEC = 10: ieee NaN arithmetic can be trusted not to trap
	//
	//     ILAENV = 0
	ilaenvReturn = 1
	if ilaenvReturn == 1 {
		ilaenvReturn = Ieeeck(1, 0.0, 1.0)
	}
	return

label150:
	;

	//     ISPEC = 11: infinity arithmetic can be trusted not to trap
	//
	//     ILAENV = 0
	ilaenvReturn = 1
	if ilaenvReturn == 1 {
		ilaenvReturn = Ieeeck(0, 0.0, 1.0)
	}
	return

label160:
	;

	//     12 <= ISPEC <= 16: xHSEQR or related subroutines.
	ilaenvReturn = Iparmq(ispec, name, opts, n1, n2, n3, n4)
	return
}
