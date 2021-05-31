package golapack

import (
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// DsytrdSb2st reduces a real symmetric band matrix A to real symmetric
// tridiagonal form T by a orthogonal similarity transformation:
// Q**T * A * Q = T.
func DsytrdSb2st(stage1, vect, uplo byte, n, kd *int, ab *mat.Matrix, ldab *int, d, e, hous *mat.Vector, lhous *int, work *mat.Vector, lwork, info *int) {
	var afters1, lquery, upper, wantq bool
	var rzero, zero float64
	var abdpos, abofdpos, apos, awpos, blklastind, colpt, dpos, ed, edind, grsiz, i, ib, inda, indtau, indv, indw, k, lda, ldv, lhmin, lwmin, m, myid, ofdpos, shift, sizea, sizetau, st, stepercol, stind, stt, sweepid, thed, thgrid, thgrnb, thgrsiz, tid, ttype int

	rzero = 0.0
	zero = 0.0

	//     Determine the minimal workspace size required.
	//     Test the input parameters
	(*info) = 0
	afters1 = stage1 == 'Y'
	wantq = vect == 'V'
	upper = uplo == 'U'
	lquery = ((*lwork) == -1) || ((*lhous) == -1)

	//     Determine the block size, the workspace size and the hous size.
	ib = Ilaenv2stage(func() *int { y := 2; return &y }(), []byte("DSYTRD_SB2ST"), []byte{vect}, n, kd, toPtr(-1), toPtr(-1))
	lhmin = Ilaenv2stage(func() *int { y := 3; return &y }(), []byte("DSYTRD_SB2ST"), []byte{vect}, n, kd, &ib, toPtr(-1))
	lwmin = Ilaenv2stage(func() *int { y := 4; return &y }(), []byte("DSYTRD_SB2ST"), []byte{vect}, n, kd, &ib, toPtr(-1))

	if !afters1 && stage1 != 'N' {
		(*info) = -1
	} else if vect != 'N' {
		(*info) = -2
	} else if !upper && uplo != 'L' {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*kd) < 0 {
		(*info) = -5
	} else if (*ldab) < ((*kd) + 1) {
		(*info) = -7
	} else if (*lhous) < lhmin && !lquery {
		(*info) = -11
	} else if (*lwork) < lwmin && !lquery {
		(*info) = -13
	}

	if (*info) == 0 {
		hous.Set(0, float64(lhmin))
		work.Set(0, float64(lwmin))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYTRD_SB2ST"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		hous.Set(0, 1)
		work.Set(0, 1)
		return
	}

	//     Determine pointer position
	ldv = (*kd) + ib
	sizetau = 2 * (*n)
	indtau = 1
	indv = indtau + sizetau
	lda = 2*(*kd) + 1
	sizea = lda * (*n)
	inda = 1
	indw = inda + sizea
	tid = 0

	if upper {
		apos = inda + (*kd)
		awpos = inda
		dpos = apos + (*kd)
		ofdpos = dpos - 1
		abdpos = (*kd) + 1
		abofdpos = (*kd)
	} else {
		apos = inda
		awpos = inda + (*kd) + 1
		dpos = apos
		ofdpos = dpos + 1
		abdpos = 1
		abofdpos = 2
	}

	//     Case KD=0:
	//     The matrix is diagonal. We just copy it (convert to "real" for
	//     real because D is double and the imaginary part should be 0)
	//     and store it in D. A sequential code here is better or
	//     in a parallel environment it might need two cores for D and E
	if (*kd) == 0 {
		for i = 1; i <= (*n); i++ {
			d.Set(i-1, ab.Get(abdpos-1, i-1))
		}
		for i = 1; i <= (*n)-1; i++ {
			e.Set(i-1, rzero)
		}

		hous.Set(0, 1)
		work.Set(0, 1)
		return
	}

	//     Case KD=1:
	//     The matrix is already Tridiagonal. We have to make diagonal
	//     and offdiagonal elements real, and store them in D and E.
	//     For that, for real precision just copy the diag and offdiag
	//     to D and E while for the COMPLEX case the bulge chasing is
	//     performed to convert the hermetian tridiagonal to symmetric
	//     tridiagonal. A simpler coversion formula might be used, but then
	//     updating the Q matrix will be required and based if Q is generated
	//     or not this might complicate the story.
	if (*kd) == 1 {
		for i = 1; i <= (*n); i++ {
			d.Set(i-1, ab.Get(abdpos-1, i-1))
		}

		if upper {
			for i = 1; i <= (*n)-1; i++ {
				e.Set(i-1, ab.Get(abofdpos-1, i+1-1))
			}
		} else {
			for i = 1; i <= (*n)-1; i++ {
				e.Set(i-1, ab.Get(abofdpos-1, i-1))
			}
		}

		hous.Set(0, 1)
		work.Set(0, 1)
		return
	}

	//     Main code start here.
	//     Reduce the symmetric band of A to a tridiagonal matrix.
	thgrsiz = (*n)
	grsiz = 1
	shift = 3
	stepercol = int(math.Ceil(float64(shift) / float64(grsiz)))
	thgrnb = int(math.Ceil(float64((*n)-1) / float64(thgrsiz)))

	Dlacpy('A', toPtr((*kd)+1), n, ab, ldab, work.MatrixOff(apos-1, lda, opts), &lda)
	Dlaset('A', kd, n, &zero, &zero, work.MatrixOff(awpos-1, lda, opts), &lda)

	//     main bulge chasing loop
	for thgrid = 1; thgrid <= thgrnb; thgrid++ {
		stt = (thgrid-1)*thgrsiz + 1
		thed = minint(stt+thgrsiz-1, (*n)-1)
		for i = stt; i <= (*n)-1; i++ {
			ed = minint(i, thed)
			if stt > ed {
				break
			}
			for m = 1; m <= stepercol; m++ {
				st = stt
				for sweepid = st; sweepid <= ed; sweepid++ {
					for k = 1; k <= grsiz; k++ {
						myid = (i-sweepid)*(stepercol*grsiz) + (m-1)*grsiz + k
						if myid == 1 {
							ttype = 1
						} else {
							ttype = (myid % 2) + 2
						}
						if ttype == 2 {
							colpt = (myid/2)*(*kd) + sweepid
							stind = colpt - (*kd) + 1
							edind = minint(colpt, *n)
							blklastind = colpt
						} else {
							colpt = ((myid+1)/2)*(*kd) + sweepid
							stind = colpt - (*kd) + 1
							edind = minint(colpt, *n)
							if (stind >= edind-1) && (edind == (*n)) {
								blklastind = (*n)
							} else {
								blklastind = 0
							}
						}

						//                         Call the kernel
						Dsb2stkernels(uplo, wantq, &ttype, &stind, &edind, &sweepid, n, kd, &ib, work.MatrixOff(inda-1, lda, opts), &lda, hous.Off(indv-1), hous.Off(indtau-1), &ldv, work.Off(indw+tid*(*kd)-1))

						if blklastind >= ((*n) - 1) {
							stt = stt + 1
							break
						}
					}
				}
			}
		}
	}

	//     Copy the diagonal from A to D. Note that D is REAL thus only
	//     the Real part is needed, the imaginary part should be zero.
	for i = 1; i <= (*n); i++ {
		d.Set(i-1, work.Get(dpos+(i-1)*lda-1))
	}

	//     Copy the off diagonal from A to E. Note that E is REAL thus only
	//     the Real part is needed, the imaginary part should be zero.
	if upper {
		for i = 1; i <= (*n)-1; i++ {
			e.Set(i-1, work.Get(ofdpos+i*lda-1))
		}
	} else {
		for i = 1; i <= (*n)-1; i++ {
			e.Set(i-1, work.Get(ofdpos+(i-1)*lda-1))
		}
	}

	hous.Set(0, float64(lhmin))
	work.Set(0, float64(lwmin))
}
