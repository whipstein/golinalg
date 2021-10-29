package golapack

// Iparam2stage This program sets problem and machine dependent parameters
//      useful for xHETRD_2STAGE, xHETRD_HE2HB, xHETRD_HB2ST,
//      xGEBRD_2STAGE, xGEBRD_GE2GB, xGEBRD_GB2BD
//      and related subroutines for eigenvalue problems.
//      It is called whenever ILAENV is called with 17 <= ISPEC <= 21.
//      It is called whenever ILAENV2STAGE is called with 1 <= ISPEC <= 5
//      with a direct conversion ISPEC + 16.
func Iparam2stage(ispec int, name string, opts []byte, ni, nbi, ibi, nxi int) (iparam2stageReturn int) {
	var cprec, rprec bool
	var vect byte
	var factoptnb, i, ib, ic, iz, kd, lhous, lqoptnb, lwork, nthreads, qroptnb int
	subnam := []byte(name)
	prec := name[0]
	algo := name[3:6]
	stag := name[6:]

	//     Invalid value for ISPEC
	if (ispec < 17) || (ispec > 21) {
		iparam2stageReturn = -1
		return
	}

	//     Get the number of threads
	nthreads = 1

	if ispec != 19 {
		//        Convert NAME to upper case if the first character is lower case.
		iparam2stageReturn = -1
		ic = int(subnam[0])
		iz = int('Z')
		if iz == 90 || iz == 122 {
			//           ASCII character set
			if ic >= 97 && ic <= 122 {
				subnam[0] = byte(ic - 32)
				for i = 2; i <= 12; i++ {
					ic = int(subnam[i-1])
					if ic >= 97 && ic <= 122 {
						subnam[i-1] = byte(ic - 32)
					}
				}
			}

		} else if iz == 233 || iz == 169 {
			//           EBCDIC character set
			if (ic >= 129 && ic <= 137) || (ic >= 145 && ic <= 153) || (ic >= 162 && ic <= 169) {
				subnam[0] = byte(ic + 64)
				for i = 2; i <= 12; i++ {
					ic = int(subnam[i-1])
					if (ic >= 129 && ic <= 137) || (ic >= 145 && ic <= 153) || (ic >= 162 && ic <= 169) {
						subnam[i-1] = byte(ic + 64)
					}
				}
			}

		} else if iz == 218 || iz == 250 {
			//           Prime machines:  ASCII+128
			if ic >= 225 && ic <= 250 {
				subnam[0] = byte(ic - 32)
				for i = 2; i <= 12; i++ {
					ic = int(subnam[i-1])
					if ic >= 225 && ic <= 250 {
						subnam[i-1] = byte(ic - 32)
					}
				}
			}
		}

		rprec = prec == 'S' || prec == 'D'
		cprec = prec == 'C' || prec == 'Z'

		//        Invalid value for PRECISION
		if !(rprec || cprec) {
			iparam2stageReturn = -1
			return
		}
	}

	if (ispec == 17) || (ispec == 18) {
		//     ISPEC = 17, 18:  block size KD, IB
		//     Could be also dependent from N but for now it
		//     depend only on sequential or parallel
		if nthreads > 4 {
			if cprec {
				kd = 128
				ib = 32
			} else {
				kd = 160
				ib = 40
			}
		} else if nthreads > 1 {
			if cprec {
				kd = 64
				ib = 32
			} else {
				kd = 64
				ib = 32
			}
		} else {
			if cprec {
				kd = 16
				ib = 16
			} else {
				kd = 32
				ib = 16
			}
		}
		if ispec == 17 {
			iparam2stageReturn = kd
		}
		if ispec == 18 {
			iparam2stageReturn = ib
		}

	} else if ispec == 19 {
		//     ISPEC = 19:
		//     LHOUS length of the Houselholder representation
		//     matrix (V,T) of the second stage. should be >= 1.
		//
		//     Will add the VECT OPTION HERE next release
		vect = opts[0]
		if vect == 'N' {
			lhous = max(1, 4*ni)
		} else {
			//           This is not correct, it need to call the ALGO and the stage2
			lhous = max(1, 4*ni) + ibi
		}
		if lhous >= 0 {
			iparam2stageReturn = lhous
		} else {
			iparam2stageReturn = -1
		}

	} else if ispec == 20 {
		//     ISPEC = 20: (21 for future use)
		//     LWORK length of the workspace for
		//     either or both stages for TRD and BRD. should be >= 1.
		//     TRD:
		//     TRD_stage 1: = LT + LW + LS1 + LS2
		//                  = LDT*KD + N*KD + N*MAX(KD,FACTOPTNB) + LDS2*KD
		//                    where LDT=LDS2=KD
		//                  = N*KD + N*max(KD,FACTOPTNB) + 2*KD*KD
		//     TRD_stage 2: = (2NB+1)*N + KD*NTHREADS
		//     TRD_both   : = max(stage1,stage2) + AB ( AB=(KD+1)*N )
		//                  = N*KD + N*max(KD+1,FACTOPTNB)
		//                    + max(2*KD*KD, KD*NTHREADS)
		//                    + (KD+1)*N
		lwork = -1
		subnam[0] = prec
		qroptnb = Ilaenv(1, string(prec)+"geqrf"+name[6:], []byte(" "), ni, nbi, -1, -1)
		lqoptnb = Ilaenv(1, string(prec)+"gelqf"+name[6:], []byte(" "), nbi, ni, -1, -1)
		//        Could be QR or LQ for TRD and the max for BRD
		factoptnb = max(qroptnb, lqoptnb)
		if algo == "trd" {
			if stag == "2stag" {
				lwork = ni*nbi + ni*max(nbi+1, factoptnb) + max(2*nbi*nbi, nbi*nthreads) + (nbi+1)*ni
			} else if (stag == "He2hb") || (stag == "Sy2sb") {
				lwork = ni*nbi + ni*max(nbi, factoptnb) + 2*nbi*nbi
			} else if (stag == "Hb2st") || (stag == "Sb2st") {
				lwork = (2*nbi+1)*ni + nbi*nthreads
			}
		} else if algo == "brd" {
			if stag == "2stag" {
				lwork = 2*ni*nbi + ni*max(nbi+1, factoptnb) + max(2*nbi*nbi, nbi*nthreads) + (nbi+1)*ni
			} else if stag == "Ge2gb" {
				lwork = ni*nbi + ni*max(nbi, factoptnb) + 2*nbi*nbi
			} else if stag == "Gb2bd" {
				lwork = (3*nbi+1)*ni + nbi*nthreads
			}
		}
		lwork = max(1, lwork)
		if lwork > 0 {
			iparam2stageReturn = lwork
		} else {
			iparam2stageReturn = -1
		}

	} else if ispec == 21 {
		//     ISPEC = 21 for future use
		iparam2stageReturn = nxi
	}

	return
}
