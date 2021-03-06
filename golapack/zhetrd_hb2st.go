package golapack

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhetrdhb2st reduces a complex Hermitian band matrix A to real symmetric
// tridiagonal form T by a unitary similarity transformation:
// Q**H * A * Q = T.
func ZhetrdHb2st(stage1, vect byte, uplo mat.MatUplo, n, kd int, ab *mat.CMatrix, d, e *mat.Vector, hous *mat.CVector, lhous int, work *mat.CVector, lwork int) (err error) {
	var afters1, lquery, upper, wantq bool
	var one, tmp, zero complex128
	var abstmp, rzero float64
	var abdpos, abofdpos, apos, awpos, blklastind, colpt, dpos, ed, edind, grsiz, i, ib, inda, indtau, indv, indw, k, lda, ldv, lhmin, lwmin, m, myid, ofdpos, shift, sizea, sizetau, st, stepercol, stind, stt, sweepid, thed, thgrid, thgrnb, thgrsiz, tid, ttype int

	rzero = 0.0
	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Determine the minimal workspace size required.
	//     Test the input parameters
	// debug = 0
	afters1 = stage1 == 'Y'
	wantq = vect == 'V'
	upper = uplo == Upper
	lquery = (lwork == -1) || (lhous == -1)

	//     Determine the block size, the workspace size and the hous size.
	ib = Ilaenv2stage(2, "ZhetrdHb2st", []byte{vect}, n, kd, -1, -1)
	lhmin = Ilaenv2stage(3, "ZhetrdHb2st", []byte{vect}, n, kd, ib, -1)
	lwmin = Ilaenv2stage(4, "ZhetrdHb2st", []byte{vect}, n, kd, ib, -1)

	if !afters1 && stage1 != 'N' {
		err = fmt.Errorf("!afters1 && stage1 != 'N': stage1='%c'", stage1)
	} else if vect != 'N' {
		err = fmt.Errorf("vect != 'N': vect='%c'", vect)
	} else if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if kd < 0 {
		err = fmt.Errorf("kd < 0: kd=%v", kd)
	} else if ab.Rows < (kd + 1) {
		err = fmt.Errorf("ab.Rows < (kd + 1): ab.Rows=%v, kd=%v", ab.Rows, kd)
	} else if lhous < lhmin && !lquery {
		err = fmt.Errorf("lhous < lhmin && !lquery: lhous=%v, lhmin=%v, lquery=%v", lhous, lhmin, lquery)
	} else if lwork < lwmin && !lquery {
		err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
	}

	if err == nil {
		hous.SetRe(0, float64(lhmin))
		work.SetRe(0, float64(lwmin))
	}

	if err != nil {
		gltest.Xerbla2("ZhetrdHb2st", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		hous.Set(0, 1)
		work.Set(0, 1)
		return
	}

	//     Determine pointer position
	ldv = kd + ib
	sizetau = 2 * n
	// sizev = 2 * n
	indtau = 1
	indv = indtau + sizetau
	lda = 2*kd + 1
	sizea = lda * n
	inda = 1
	indw = inda + sizea
	// nthreads = 1
	tid = 0

	if upper {
		apos = inda + kd
		awpos = inda
		dpos = apos + kd
		ofdpos = dpos - 1
		abdpos = kd + 1
		abofdpos = kd
	} else {
		apos = inda
		awpos = inda + kd + 1
		dpos = apos
		ofdpos = dpos + 1
		abdpos = 1
		abofdpos = 2
	}

	//     Case KD=0:
	//     The matrix is diagonal. We just copy it (convert to "real" for
	//     complex because D is double and the imaginary part should be 0)
	//     and store it in D. A sequential code here is better or
	//     in a parallel environment it might need two cores for D and E
	if kd == 0 {
		for i = 1; i <= n; i++ {
			d.Set(i-1, ab.GetRe(abdpos-1, i-1))
		}
		for i = 1; i <= n-1; i++ {
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
	if kd == 1 {
		for i = 1; i <= n; i++ {
			d.Set(i-1, ab.GetRe(abdpos-1, i-1))
		}

		//         make off-diagonal elements real and copy them to E
		if upper {
			for i = 1; i <= n-1; i++ {
				tmp = ab.Get(abofdpos-1, i)
				abstmp = cmplx.Abs(tmp)
				ab.SetRe(abofdpos-1, i, abstmp)
				e.Set(i-1, abstmp)
				if abstmp != rzero {
					tmp = tmp / complex(abstmp, 0)
				} else {
					tmp = one
				}
				if i < n-1 {
					ab.Set(abofdpos-1, i+2-1, ab.Get(abofdpos-1, i+2-1)*tmp)
				}
			}
		} else {
			for i = 1; i <= n-1; i++ {
				tmp = ab.Get(abofdpos-1, i-1)
				abstmp = cmplx.Abs(tmp)
				ab.SetRe(abofdpos-1, i-1, abstmp)
				e.Set(i-1, abstmp)
				if abstmp != rzero {
					tmp = tmp / complex(abstmp, 0)
				} else {
					tmp = one
				}
				if i < n-1 {
					ab.Set(abofdpos-1, i, ab.Get(abofdpos-1, i)*tmp)
				}
			}
		}

		hous.Set(0, 1)
		work.Set(0, 1)
		return
	}

	//     Main code start here.
	//     Reduce the hermitian band of A to a tridiagonal matrix.
	thgrsiz = n
	grsiz = 1
	shift = 3
	// nbtiles = int(math.Ceil(float64n / float64kd))
	stepercol = int(math.Ceil(float64(shift) / float64(grsiz)))
	thgrnb = int(math.Ceil(float64(n-1) / float64(thgrsiz)))

	Zlacpy(Full, kd+1, n, ab, work.Off(apos-1).CMatrix(lda, opts))
	Zlaset(Full, kd, n, zero, zero, work.Off(awpos-1).CMatrix(lda, opts))

	//     main bulge chasing loop
	for thgrid = 1; thgrid <= thgrnb; thgrid++ {
		stt = (thgrid-1)*thgrsiz + 1
		thed = min(stt+thgrsiz-1, n-1)
		for i = stt; i <= n-1; i++ {
			ed = min(i, thed)
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
							colpt = (myid/2)*kd + sweepid
							stind = colpt - kd + 1
							edind = min(colpt, n)
							blklastind = colpt
						} else {
							colpt = ((myid+1)/2)*kd + sweepid
							stind = colpt - kd + 1
							edind = min(colpt, n)
							if (stind >= edind-1) && (edind == n) {
								blklastind = n
							} else {
								blklastind = 0
							}
						}

						Zhb2stKernels(uplo, wantq, ttype, stind, edind, sweepid, n, kd, ib, work.Off(inda-1).CMatrix(lda, opts), hous.Off(indv-1), hous.Off(indtau-1), ldv, work.Off(indw+tid*kd-1))

						if blklastind >= (n - 1) {
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
	for i = 1; i <= n; i++ {
		d.Set(i-1, work.GetRe(dpos+(i-1)*lda-1))
	}

	//     Copy the off diagonal from A to E. Note that E is REAL thus only
	//     the Real part is needed, the imaginary part should be zero.
	if upper {
		for i = 1; i <= n-1; i++ {
			e.Set(i-1, work.GetRe(ofdpos+i*lda-1))
		}
	} else {
		for i = 1; i <= n-1; i++ {
			e.Set(i-1, work.GetRe(ofdpos+(i-1)*lda-1))
		}
	}

	hous.SetRe(0, float64(lhmin))
	work.SetRe(0, float64(lwmin))

	return
}
