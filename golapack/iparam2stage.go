package golapack

// Iparam2stage This program sets problem and machine dependent parameters
//      useful for xHETRD_2STAGE, xHETRD_HE2HB, xHETRD_HB2ST,
//      xGEBRD_2STAGE, xGEBRD_GE2GB, xGEBRD_GB2BD
//      and related subroutines for eigenvalue problems.
//      It is called whenever ILAENV is called with 17 <= ISPEC <= 21.
//      It is called whenever ILAENV2STAGE is called with 1 <= ISPEC <= 5
//      with a direct conversion ISPEC + 16.
func Iparam2stage(ispec *int, name, opts []byte, ni, nbi, ibi, nxi *int) (iparam2stageReturn int) {
	var cprec, rprec bool
	var prec, vect byte
	var factoptnb, i, ib, ic, iz, kd, lhous, lqoptnb, lwork, nthreads, qroptnb int
	subnam := []byte(name)
	prec = subnam[0]
	algo := []byte(subnam[3:6])
	stag := []byte(subnam[7:12])

	//! #if defined(_OPENMP)
	//!       use omp_lib
	//! #endif

	//     Invalid value for ISPEC
	if ((*ispec) < 17) || ((*ispec) > 21) {
		iparam2stageReturn = -1
		return
	}

	//     Get the number of threads
	nthreads = 1
	//! #if defined(_OPENMP)
	//! !$OMP PARALLEL
	//!       NTHREADS = OMP_GET_NUM_THREADS()
	//! !$OMP END PARALLEL
	//! #endif
	//      WRITE(*,*) 'IPARAM VOICI NTHREADS ISPEC ',NTHREADS, ISPEC

	if (*ispec) != 19 {
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
	//      WRITE(*,*),'RPREC,CPREC ',RPREC,CPREC,
	//     $           '   ALGO ',ALGO,'    STAGE ',STAG

	if ((*ispec) == 17) || ((*ispec) == 18) {
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
		if (*ispec) == 17 {
			iparam2stageReturn = kd
		}
		if (*ispec) == 18 {
			iparam2stageReturn = ib
		}

	} else if (*ispec) == 19 {
		//     ISPEC = 19:
		//     LHOUS length of the Houselholder representation
		//     matrix (V,T) of the second stage. should be >= 1.
		//
		//     Will add the VECT OPTION HERE next release
		vect = opts[0]
		if vect == 'N' {
			lhous = maxint(1, 4*(*ni))
		} else {
			//           This is not correct, it need to call the ALGO and the stage2
			lhous = maxint(1, 4*(*ni)) + (*ibi)
		}
		if lhous >= 0 {
			iparam2stageReturn = lhous
		} else {
			iparam2stageReturn = -1
		}

	} else if (*ispec) == 20 {
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
		qroptnb = Ilaenv(func() *int { y := 1; return &y }(), []byte(string(prec)+"GEQRF"+string(subnam[6:])), []byte(" "), ni, nbi, toPtr(-1), toPtr(-1))
		lqoptnb = Ilaenv(func() *int { y := 1; return &y }(), []byte(string(prec)+"GELQF"+string(subnam[6:])), []byte(" "), nbi, ni, toPtr(-1), toPtr(-1))
		//        Could be QR or LQ for TRD and the max for BRD
		factoptnb = maxint(qroptnb, lqoptnb)
		if string(algo) == "TRD" {
			if string(stag) == "2STAG" {
				lwork = (*ni)*(*nbi) + (*ni)*maxint((*nbi)+1, factoptnb) + maxint(2*(*nbi)*(*nbi), (*nbi)*nthreads) + ((*nbi)+1)*(*ni)
			} else if (string(stag) == "HE2HB") || (string(stag) == "SY2SB") {
				lwork = (*ni)*(*nbi) + (*ni)*maxint(*nbi, factoptnb) + 2*(*nbi)*(*nbi)
			} else if (string(stag) == "HB2ST") || (string(stag) == "SB2ST") {
				lwork = (2*(*nbi)+1)*(*ni) + (*nbi)*nthreads
			}
		} else if string(algo) == "BRD" {
			if string(stag) == "2STAG" {
				lwork = 2*(*ni)*(*nbi) + (*ni)*maxint((*nbi)+1, factoptnb) + maxint(2*(*nbi)*(*nbi), (*nbi)*nthreads) + ((*nbi)+1)*(*ni)
			} else if string(stag) == "GE2GB" {
				lwork = (*ni)*(*nbi) + (*ni)*maxint(*nbi, factoptnb) + 2*(*nbi)*(*nbi)
			} else if string(stag) == "GB2BD" {
				lwork = (3*(*nbi)+1)*(*ni) + (*nbi)*nthreads
			}
		}
		lwork = maxint(1, lwork)
		if lwork > 0 {
			iparam2stageReturn = lwork
		} else {
			iparam2stageReturn = -1
		}

	} else if (*ispec) == 21 {
		//     ISPEC = 21 for future use
		iparam2stageReturn = (*nxi)
	}

	return
}
