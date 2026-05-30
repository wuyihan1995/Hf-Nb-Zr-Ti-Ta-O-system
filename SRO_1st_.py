import numpy as np
from math import sqrt
import mpi4py.MPI as MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


# =========================================================================
#  ONLY EDIT THIS BLOCK TO STUDY A DIFFERENT SYSTEM
# -------------------------------------------------------------------------
#  ElemNames: element symbols ordered by LAMMPS atom type (type 1, 2, 3, ...)
#             e.g. for a Mo-Nb-Ta-W system use ["Mo", "Nb", "Ta", "W"]
#  STR      : crystal structure, "BCC" (CN=8) or "FCC" (CN=12)
# =========================================================================
ElemNames = ["W", 'Ta', 'Cr', 'V']
STR       = "BCC"
fnformer  = "dump."
fnlatter  = "300000"
DisLimit  = 4
# =========================================================================

AtomType = len(ElemNames)
filename = fnformer + fnlatter

if STR == "BCC":
    CoordNum = 8
elif STR == "FCC":
    CoordNum = 12
else:
    raise ValueError("STR must be 'BCC' or 'FCC'")


# ============================  Delta function  ===========================
def deltaFunc(a, b):
    return 1 if a == b else 0


# ============================  Bubble sort  ==============================
def Ranking(A):
    row, col = np.shape(A)
    for i in range(1, row):
        for j in range(0, row - i):
            if A[j, -1] >= A[j + 1, -1]:
                temp = A[j, :].copy()
                A[j, :] = A[j + 1, :].copy()
                A[j + 1, :] = temp
    return A


# ====================  Per-element SRO calculation  =====================
def calc_sro_for_element(CurrentType, list_temp, AtomList, Num, box,
                         CoordNum, AtomType, DisLimit, comm):
    """Compute Warren-Cowley SRO parameters for one element type.

    Returns the alpha vector (length AtomType) on rank 0, None elsewhere.
    Generalized over an arbitrary number of element types: per-type
    neighbour counts are accumulated in the array P[AtomType].
    """
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    xlo, xhi, ylo, yhi, zlo, zhi = box

    NumThis   = int(Num[int(CurrentType - 1)])
    GroupSize = NumThis // comm_size

    # last rank takes the remainder, others take an even slice
    if comm_rank != comm_size - 1:
        i_start = comm_rank * GroupSize
        i_end   = (comm_rank + 1) * GroupSize
    else:
        i_start = comm_rank * GroupSize
        i_end   = NumThis

    P = np.zeros(AtomType)

    for i in range(i_start, i_end):
        POSi = list_temp[i, :]
        NeighList = np.ones((1, 2)) * 10.0
        CountTmp = 0
        for j in range(0, AtomNum):
            POSj = AtomList[j, 1:]
            # minimum-image convention; x/y are periodic, z keeps the same
            # treatment because the vacuum layer (>> cutoff) makes the
            # z-mirror term always larger than the cutoff and thus harmless
            ddx = min(abs(POSi[0] - POSj[0]), (xhi - xlo) - abs(POSi[0] - POSj[0]))
            ddy = min(abs(POSi[1] - POSj[1]), (yhi - ylo) - abs(POSi[1] - POSj[1]))
            ddz = min(abs(POSi[2] - POSj[2]), (zhi - zlo) - abs(POSi[2] - POSj[2]))
            R = sqrt(ddx * ddx + ddy * ddy + ddz * ddz)

            if 0.05 < R < DisLimit:
                NeighList[CountTmp, :] = [AtomList[j, 0], R]
                CountTmp += 1
                NeighList = np.insert(NeighList, CountTmp, 10, axis=0)

        NearestList = Ranking(NeighList)[0:CoordNum, :]
        for j in range(0, np.shape(NearestList)[0]):
            # skip placeholder rows (padding) whose distance is the sentinel
            # value (>= DisLimit); only count genuine neighbours
            if NearestList[j, 1] >= DisLimit:
                continue
            ntype = int(NearestList[j, 0])
            if 1 <= ntype <= AtomType:
                P[ntype - 1] += 1

        if (i + 1) % 100 == 0:
            print("        ", "Proc #", comm_rank, "finish ", str(int(i + 1)), " atoms")

    P = comm.gather(P, root=0)

    if comm_rank == 0:
        S = np.sum(P, axis=0)            # total neighbour counts per type
        S = S / CoordNum / Num[int(CurrentType - 1)]
        delta = np.array([deltaFunc(t, CurrentType) for t in range(1, AtomType + 1)])
        alpha = (S - 1.0 / AtomType) / (delta - 1.0 / AtomType)
        return alpha
    return None


# ===============================  Main  =================================
if __name__ == "__main__":

    # -----------------------  Read & store atoms (rank 0)  --------------
    Lists = None
    if comm_rank == 0:
        print("Reading file and store list ...")
        print(" ")
        with open(filename, "r") as file:
            line = 0
            for lines in file.readlines():
                line += 1
                if line == 4:
                    AtomNum  = int(lines)
                    AtomList = np.zeros((AtomNum, 4))
                    Lists    = [np.zeros((1, 3)) for _ in range(AtomType)]
                    Num      = np.zeros((AtomType,))
                if line == 6:
                    LINE = list(map(float, lines.strip().split()))
                    xlo, xhi = LINE[0], LINE[1]
                if line == 7:
                    LINE = list(map(float, lines.strip().split()))
                    ylo, yhi = LINE[0], LINE[1]
                if line == 8:
                    LINE = list(map(float, lines.strip().split()))
                    zlo, zhi = LINE[0], LINE[1]
                if line >= 10:
                    LINE = list(map(float, lines.strip().split()))
                    idx = int(np.sum(Num))
                    AtomList[idx, 0] = LINE[1]
                    AtomList[idx, 1] = LINE[2]
                    AtomList[idx, 2] = LINE[3]
                    AtomList[idx, 3] = LINE[4]

                    t = int(LINE[1])
                    if 1 <= t <= AtomType:
                        slot = int(Num[t - 1])
                        Lists[t - 1][slot, :] = [LINE[2], LINE[3], LINE[4]]
                        Num[t - 1] += 1
                        Lists[t - 1] = np.insert(Lists[t - 1], int(Num[t - 1]), 0, axis=0)

    # ---------------------------  Broadcast  ----------------------------
    AtomList = comm.bcast(AtomList if comm_rank == 0 else None, root=0)
    Lists    = comm.bcast(Lists    if comm_rank == 0 else None, root=0)

    xlo = comm.bcast(xlo if comm_rank == 0 else None, root=0)
    xhi = comm.bcast(xhi if comm_rank == 0 else None, root=0)   # FIX: was xlo
    ylo = comm.bcast(ylo if comm_rank == 0 else None, root=0)
    yhi = comm.bcast(yhi if comm_rank == 0 else None, root=0)
    zlo = comm.bcast(zlo if comm_rank == 0 else None, root=0)
    zhi = comm.bcast(zhi if comm_rank == 0 else None, root=0)

    AtomNum = comm.bcast(AtomNum if comm_rank == 0 else None, root=0)
    Num     = comm.bcast(Num     if comm_rank == 0 else None, root=0)

    box = (xlo, xhi, ylo, yhi, zlo, zhi)

    # ----------------------  Loop over all elements  --------------------
    for t in range(1, AtomType + 1):
        if comm_rank == 0:
            print("    Calculating SRO for Element  #", t, "(", ElemNames[t - 1], ") ...")

        alpha = calc_sro_for_element(t, Lists[t - 1], AtomList, Num, box,
                                     CoordNum, AtomType, DisLimit, comm)

        if comm_rank == 0:
            header = ", ".join(str(k) for k in range(1, AtomType + 1))
            np.savetxt(ElemNames[t - 1] + "_SRO_1st.txt", alpha, header=header)
